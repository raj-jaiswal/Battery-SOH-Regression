#!/usr/bin/env python3
"""Optimized battery SOH inference + hardware benchmark runtime for Jetson Orin Nano.

Input contract:
  data/unified.parquet is already cycle-level. Exactly one row per discharge cycle.
  This runtime does NOT aggregate raw telemetry and does NOT load training joblibs.

Model artifacts:
  models/optimized/<model_name>/metadata.json + model-format-specific files.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_NAMES = [
    "gpr_rbf", "gpr_matern", "gpr_ard", "sparse_gpr",
    "deep_kernel_gpr", "svr", "random_forest", "xgboost",
    "wide_nn", "deep_nn", "wide_deep_nn", "bilstm",
    "symbolic_regression",
]
TARGET = "soh_pct"
SEQUENCE_LENGTH_DEFAULT = 8


def find_project_root(script_path: str | Path) -> Path:
    here = Path(script_path).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "data" / "unified.parquet").is_file() and (candidate / "models" / "optimized").is_dir():
            return candidate
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "data" / "unified.parquet").is_file() and (candidate / "models" / "optimized").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find project root containing data/unified.parquet and models/optimized/. "
        "Run the runner from inside PBL/ or preserve the PBL folder structure."
    )


def is_jetson() -> bool:
    return Path("/etc/nv_tegra_release").exists() or (
        platform.machine().lower() in {"aarch64", "arm64"} and shutil.which("tegrastats") is not None
    )


def _safe_json_load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metadata file: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return obj


def load_cycle_frame(parquet_path: Path, metadata: dict) -> pd.DataFrame:
    """Load the exact cycle-level predictors needed by the trained model."""
    numeric = list(metadata.get("numeric_features", []))
    categorical = list(metadata.get("categorical_features", []))
    feature_cols = list(metadata.get("feature_columns", []))
    if feature_cols != numeric + categorical:
        raise ValueError("metadata feature_columns order does not match numeric + categorical features.")

    required = {"source_dataset", "entity_id", "cycle_id", "cycle_order"} | set(feature_cols)
    try:
        available = set(pd.read_parquet(parquet_path, engine="pyarrow").columns)
    except Exception as exc:
        raise RuntimeError(f"Could not read {parquet_path} as genuine Parquet.") from exc
    missing = required - available
    if missing:
        raise ValueError(f"unified.parquet is missing required fields: {sorted(missing)}")

    columns = list(dict.fromkeys(required))
    frame = pd.read_parquet(parquet_path, columns=columns, engine="pyarrow")
    frame["source_dataset"] = frame["source_dataset"].astype("string").str.strip().str.upper()
    frame = frame.loc[frame["source_dataset"].isin(["NASA", "CALCE"])].copy()
    if frame.empty:
        raise ValueError("No NASA/CALCE records are available in unified.parquet.")

    for col in ("entity_id", "cycle_id"):
        frame[col] = frame[col].astype("string").str.strip()
        if frame[col].isna().any() or frame[col].astype(str).str.len().eq(0).any():
            raise ValueError(f"{col} contains missing/empty values.")

    frame["cycle_order"] = pd.to_numeric(frame["cycle_order"], errors="coerce")
    if frame["cycle_order"].isna().any() or not np.isfinite(frame["cycle_order"].to_numpy(float)).all():
        raise ValueError("cycle_order must be finite and numeric.")

    dup = frame.duplicated(["source_dataset", "entity_id", "cycle_id"], keep=False)
    if dup.any():
        example = frame.loc[dup, ["source_dataset", "entity_id", "cycle_id"]].head(10)
        raise ValueError("Duplicate cycle keys in unified.parquet:\n" + example.to_string(index=False))

    frame["cell_group"] = frame["source_dataset"].astype(str) + "::" + frame["entity_id"].astype(str)
    frame["cycle_index"] = frame["cycle_order"].astype(float)
    frame = frame.sort_values(
        ["source_dataset", "entity_id", "cycle_order", "cycle_id"], kind="mergesort"
    ).reset_index(drop=True)
    return frame


def transform_from_metadata(frame: pd.DataFrame, metadata: dict) -> np.ndarray:
    """Recreate the fitted sklearn ColumnTransformer from exported scalar metadata."""
    numeric = list(metadata.get("numeric_features", []))
    categorical = list(metadata.get("categorical_features", []))
    blocks: list[np.ndarray] = []

    num_cfg = metadata.get("numeric_preprocessing")
    if numeric:
        if not isinstance(num_cfg, dict):
            raise ValueError("Numeric preprocessing parameters are missing from metadata.json.")
        raw = np.column_stack([
            pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=np.float64)
            if c in frame else np.full(len(frame), np.nan)
            for c in numeric
        ])
        raw[~np.isfinite(raw)] = np.nan
        missing = np.isnan(raw)
        stats = np.asarray(num_cfg["imputer_statistics"], dtype=np.float64)
        if len(stats) != raw.shape[1]:
            raise ValueError(
                f"Numeric imputer expected {raw.shape[1]} columns, metadata has {len(stats)}."
            )
        imputed = np.where(missing, stats[None, :], raw)
        indicator_features = np.asarray(num_cfg.get("indicator_features", []), dtype=int)
        if indicator_features.size:
            indicators = missing[:, indicator_features].astype(np.float64)
            imputed = np.concatenate([imputed, indicators], axis=1)
        mean = np.asarray(num_cfg["scaler_mean"], dtype=np.float64)
        scale = np.asarray(num_cfg["scaler_scale"], dtype=np.float64)
        if imputed.shape[1] != len(mean) or len(mean) != len(scale):
            raise ValueError("Numeric imputer/scaler dimensions in metadata.json are inconsistent.")
        scale = np.where(scale == 0.0, 1.0, scale)
        blocks.append(((imputed - mean[None, :]) / scale[None, :]).astype(np.float32))

    cat_cfg = metadata.get("categorical_preprocessing")
    if categorical:
        if not isinstance(cat_cfg, dict):
            raise ValueError("Categorical preprocessing parameters are missing from metadata.json.")
        cat_blocks = []
        modes = cat_cfg["imputer_statistics"]
        categories = cat_cfg["categories"]
        if len(modes) != len(categorical) or len(categories) != len(categorical):
            raise ValueError("Categorical metadata dimensions do not match categorical features.")
        for j, col in enumerate(categorical):
            known = [str(v) for v in categories[j]]
            if col in frame:
                source = frame[col].to_numpy(dtype=object)
            else:
                source = np.asarray([modes[j]] * len(frame), dtype=object)
            values = []
            for value in source:
                try:
                    missing_value = value is None or bool(pd.isna(value))
                except (TypeError, ValueError):
                    missing_value = False
                if missing_value:
                    if "None" in known:
                        values.append("None")
                    else:
                        values.append(str(modes[j]))
                else:
                    values.append(str(value))
            values = np.asarray(values, dtype=object)
            # sklearn OneHotEncoder(handle_unknown='ignore') emits all zeros for unseen levels.
            if known:
                cat_blocks.append(np.column_stack([(values == category).astype(np.float32) for category in known]))
            else:
                cat_blocks.append(np.zeros((len(frame), 0), dtype=np.float32))
        if cat_blocks:
            blocks.append(np.concatenate(cat_blocks, axis=1).astype(np.float32))

    if not blocks:
        raise ValueError("No preprocessing feature blocks could be generated.")
    X = np.concatenate(blocks, axis=1).astype(np.float32)
    expected = len(metadata.get("transformed_feature_names", []))
    if X.ndim != 2 or X.shape[1] != expected:
        raise ValueError(f"Preprocessing produced shape {X.shape}; metadata expects {expected} transformed features.")
    return X


def make_sequences(frame: pd.DataFrame, X: np.ndarray, sequence_length: int) -> np.ndarray:
    """Match the notebook's left-padded causal sequence ending at each current cycle."""
    X = np.asarray(X, dtype=np.float32)
    result = np.zeros((len(frame), sequence_length, X.shape[1]), dtype=np.float32)
    for _, indices in frame.groupby("cell_group", sort=False).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        order = np.argsort(frame.iloc[positions]["cycle_index"].to_numpy(), kind="stable")
        positions = positions[order]
        for j, end_pos in enumerate(positions):
            start = max(0, j - sequence_length + 1)
            window = positions[start:j + 1]
            result[end_pos, -len(window):, :] = X[window]
    return result


def _rbf(X, Z, lengthscale, outputscale=1.0, chunk=128):
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    ls = np.asarray(lengthscale, dtype=np.float64).reshape(-1)
    if ls.size == 1:
        ls = np.repeat(ls, X.shape[1])
    if ls.size != X.shape[1]:
        raise ValueError("RBF lengthscale dimension does not match feature dimension.")
    Zs = Z / ls[None, :]
    z2 = np.einsum("ij,ij->i", Zs, Zs)[None, :]
    out = np.empty((len(X), len(Z)), dtype=np.float64)
    for start in range(0, len(X), chunk):
        Xs = X[start:start + chunk] / ls[None, :]
        x2 = np.einsum("ij,ij->i", Xs, Xs)[:, None]
        d2 = np.maximum(x2 + z2 - 2.0 * (Xs @ Zs.T), 0.0)
        out[start:start + chunk] = float(outputscale) * np.exp(-0.5 * d2)
    return out


def _matern(X, Z, lengthscale, outputscale, nu, chunk=128):
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    ls = np.asarray(lengthscale, dtype=np.float64).reshape(-1)
    if ls.size == 1:
        ls = np.repeat(ls, X.shape[1])
    if ls.size != X.shape[1]:
        raise ValueError("Matérn lengthscale dimension does not match feature dimension.")
    Zs = Z / ls[None, :]
    z2 = np.einsum("ij,ij->i", Zs, Zs)[None, :]
    out = np.empty((len(X), len(Z)), dtype=np.float64)
    for start in range(0, len(X), chunk):
        Xs = X[start:start + chunk] / ls[None, :]
        x2 = np.einsum("ij,ij->i", Xs, Xs)[:, None]
        r = np.sqrt(np.maximum(x2 + z2 - 2.0 * (Xs @ Zs.T), 0.0))
        if abs(float(nu) - 0.5) < 1e-8:
            k = np.exp(-r)
        elif abs(float(nu) - 1.5) < 1e-8:
            q = math.sqrt(3.0) * r
            k = (1.0 + q) * np.exp(-q)
        elif abs(float(nu) - 2.5) < 1e-8:
            q = math.sqrt(5.0) * r
            k = (1.0 + q + q * q / 3.0) * np.exp(-q)
        else:
            raise ValueError(f"Unsupported Matérn nu={nu}.")
        out[start:start + chunk] = float(outputscale) * k
    return out


def _predict_rf_arrays(X, arrays, return_tree_spread=False):
    # Match sklearn tree semantics: cast to float32 before threshold comparison.
    # Vectorize node traversal across rows to keep the 400-tree model practical on Orin.
    X = np.asarray(X, dtype=np.float32)
    left, right = arrays["left"], arrays["right"]
    feature, threshold, values = arrays["feature"], arrays["threshold"], arrays["value"]
    counts = arrays["node_count"]
    tree_predictions = np.empty((len(counts), len(X)), dtype=np.float64) if return_tree_spread else None
    total = np.zeros(len(X), dtype=np.float64)

    for tree_idx, node_count in enumerate(counts):
        nodes = np.zeros(len(X), dtype=np.int32)
        steps = 0
        while True:
            active_rows = np.flatnonzero(left[tree_idx, nodes] != -1)
            if active_rows.size == 0:
                break
            current_nodes = nodes[active_rows]
            feature_ids = feature[tree_idx, current_nodes].astype(np.int64)
            go_left = X[active_rows, feature_ids] <= threshold[tree_idx, current_nodes]
            nodes[active_rows] = np.where(
                go_left,
                left[tree_idx, current_nodes],
                right[tree_idx, current_nodes],
            )
            steps += 1
            if steps > int(node_count):
                raise RuntimeError("Invalid RandomForest tree array: traversal did not reach a leaf.")

        pred = values[tree_idx, nodes]
        total += pred
        if return_tree_spread:
            tree_predictions[tree_idx] = pred

    mean = total / max(1, len(counts))
    if return_tree_spread:
        return mean, tree_predictions.std(axis=0, ddof=0)
    return mean


def _eval_symbolic(node, X):
    X = np.asarray(X, dtype=np.float64)
    if "feature" in node:
        return X[:, int(node["feature"])]
    if "constant" in node:
        return np.full(len(X), float(node["constant"]), dtype=np.float64)
    args = [_eval_symbolic(arg, X) for arg in node["args"]]
    op = node["op"]
    if op == "add": return args[0] + args[1]
    if op == "sub": return args[0] - args[1]
    if op == "mul": return args[0] * args[1]
    if op == "div":
        a, b = args
        good = np.abs(b) > 0.001
        return np.where(good, a / np.where(good, b, 1.0), 1.0)
    if op == "sqrt": return np.sqrt(np.abs(args[0]))
    if op == "log":
        a = np.abs(args[0])
        good = a > 0.001
        return np.where(good, np.log(np.maximum(a, 1e-300)), 0.0)
    if op == "abs": return np.abs(args[0])
    if op == "neg": return -args[0]
    raise ValueError(f"Unsupported symbolic operator: {op!r}")


class TensorRTRunner:
    """Direct TensorRT runner supporting TensorRT 8 binding API and TensorRT 10 tensor API."""
    def __init__(self, engine_path: Path):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
        self.trt, self.cuda = trt, cuda
        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.host = {}
        self.device = {}
        self.shape_key = None
        self.new_api = hasattr(self.engine, "num_io_tensors")
        if self.new_api:
            self.inputs, self.outputs = [], []
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                (self.inputs if mode == trt.TensorIOMode.INPUT else self.outputs).append(name)
            if len(self.inputs) != 1 or len(self.outputs) != 1:
                raise RuntimeError("Expected one TensorRT input and one output.")
            self.input_key, self.output_key = self.inputs[0], self.outputs[0]
        else:
            self.input_idx = next(i for i in range(self.engine.num_bindings)
                                  if self.engine.binding_is_input(i))
            self.output_idx = next(i for i in range(self.engine.num_bindings)
                                   if not self.engine.binding_is_input(i))
            self.input_key, self.output_key = self.input_idx, self.output_idx

    def _allocate(self, X):
        if self.new_api:
            if not self.context.set_input_shape(self.input_key, tuple(X.shape)):
                raise RuntimeError(f"TensorRT rejected input shape {X.shape}.")
            keys = [self.input_key, self.output_key]
            for key in keys:
                shape = tuple(self.context.get_tensor_shape(key))
                if any(int(v) < 0 for v in shape):
                    raise RuntimeError(f"Unresolved TensorRT tensor shape {key}: {shape}")
                dtype = self.trt.nptype(self.engine.get_tensor_dtype(key))
                host = self.cuda.pagelocked_empty(int(np.prod(shape)), dtype).reshape(shape)
                device = self.cuda.mem_alloc(host.nbytes)
                self.host[key], self.device[key] = host, device
                self.context.set_tensor_address(key, int(device))
        else:
            self.context.set_binding_shape(self.input_idx, tuple(X.shape))
            for i in range(self.engine.num_bindings):
                shape = tuple(self.context.get_binding_shape(i))
                if any(int(v) < 0 for v in shape):
                    raise RuntimeError(f"Unresolved TensorRT binding shape {i}: {shape}")
                dtype = self.trt.nptype(self.engine.get_binding_dtype(i))
                host = self.cuda.pagelocked_empty(int(np.prod(shape)), dtype).reshape(shape)
                device = self.cuda.mem_alloc(host.nbytes)
                self.host[i], self.device[i] = host, device
            self.bindings = [int(self.device[i]) for i in range(self.engine.num_bindings)]
        self.shape_key = tuple(X.shape)

    def predict(self, X):
        X = np.asarray(X)
        if tuple(X.shape) != self.shape_key:
            self._allocate(X)
        key_in, key_out = self.input_key, self.output_key
        host_in = self.host[key_in]
        np.copyto(host_in, X.astype(host_in.dtype, copy=False))
        self.cuda.memcpy_htod_async(self.device[key_in], host_in, self.stream)
        if self.new_api:
            ok = self.context.execute_async_v3(self.stream.handle)
        else:
            ok = self.context.execute_async_v2(self.bindings, self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT asynchronous execution failed.")
        self.cuda.memcpy_dtoh_async(self.host[key_out], self.device[key_out], self.stream)
        self.stream.synchronize()
        return np.asarray(self.host[key_out]).reshape(len(X), -1)[:, 0].astype(np.float64)


class OnnxRunner:
    def __init__(self, onnx_path: Path, provider="auto"):
        self.backend = None
        engine_path = onnx_path.with_suffix(".engine")
        if provider in ("auto", "tensorrt") and engine_path.is_file():
            try:
                self.backend = TensorRTRunner(engine_path)
                self.provider_name = "TensorRT engine"
                return
            except Exception as exc:
                if provider == "tensorrt":
                    raise RuntimeError(f"TensorRT engine load failed: {exc}") from exc
                print(f"TensorRT engine could not load; trying ONNX Runtime: {exc}")

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                f"{onnx_path.name} needs ONNX Runtime, or a compatible .engine + TensorRT/PyCUDA."
            ) from exc

        available = ort.get_available_providers()
        providers = []
        if provider in ("auto", "tensorrt") and "TensorrtExecutionProvider" in available:
            cache = onnx_path.parent / "trt_cache"
            cache.mkdir(exist_ok=True)
            providers.append(("TensorrtExecutionProvider", {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(cache),
            }))
        if provider in ("auto", "cuda") and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        if provider == "cuda" and "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider is unavailable in this ONNX Runtime build.")
        if provider == "tensorrt" and "TensorrtExecutionProvider" not in available:
            raise RuntimeError("TensorRT provider unavailable and no compatible .engine file exists.")
        if provider in ("auto", "cpu"):
            providers.append("CPUExecutionProvider")
        if not providers:
            providers = ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        self.backend = ort.InferenceSession(str(onnx_path), sess_options=options, providers=providers)
        self.input_name = self.backend.get_inputs()[0].name
        self.output_name = self.backend.get_outputs()[0].name
        self.provider_name = ",".join(self.backend.get_providers())

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        if isinstance(self.backend, TensorRTRunner):
            return self.backend.predict(X)
        result = self.backend.run([self.output_name], {self.input_name: X})[0]
        return np.asarray(result).reshape(len(X), -1)[:, 0].astype(np.float64)


class OptimizedPredictor:
    def __init__(self, model_name: str, model_dir: Path, metadata: dict, provider="auto"):
        self.model_name = model_name
        self.model_dir = model_dir
        self.metadata = metadata
        self.model_format = metadata["model_format"]
        self.provider = "CPU/NumPy"
        self.data = None
        self.onnx = None
        self.feature_extractor = None
        self.booster = None
        self.symbolic_c = None

        if self.model_format in {
            "exact_gpr_numpy", "rbf_svr_numpy", "random_forest_arrays", "sparse_gpr_numpy"
        }:
            self.data = dict(np.load(model_dir / "model.npz", allow_pickle=False))
        elif self.model_format == "deep_kernel_gpr_numpy_plus_onnx":
            self.data = dict(np.load(model_dir / "gp_params.npz", allow_pickle=False))
            onnx_file = metadata.get("onnx_file", "feature_extractor.onnx")
            self.feature_extractor = OnnxRunner(model_dir / onnx_file, provider=provider)
            self.provider = self.feature_extractor.provider_name
        elif self.model_format == "onnx_neural":
            onnx_file = metadata.get("onnx_file", "model.onnx")
            self.onnx = OnnxRunner(model_dir / onnx_file, provider=provider)
            self.provider = self.onnx.provider_name
        elif self.model_format == "xgboost_ubjson":
            try:
                import xgboost as xgb
            except ImportError as exc:
                raise ImportError("xgboost is needed for the optimized XGBoost artifact.") from exc
            self.booster = xgb.Booster()
            self.booster.load_model(str(model_dir / metadata.get("file", "model.ubj")))
            self.provider = "XGBoost native"
        elif self.model_format == "symbolic_c_json":
            expression_file = metadata.get("expression_file", "equation.json")
            self.symbolic_tree = json.loads((model_dir / expression_file).read_text(encoding="utf-8"))
            self._load_symbolic_c()
            self.provider = "compiled C" if self.symbolic_c is not None else "NumPy expression"
        else:
            raise ValueError(f"Unsupported optimized model format: {self.model_format!r}")

    def _load_symbolic_c(self):
        c_path = self.model_dir / "equation.c"
        if not c_path.is_file() or shutil.which("gcc") is None:
            return
        so_path = self.model_dir / "equation.so"
        try:
            if not so_path.exists() or so_path.stat().st_mtime < c_path.stat().st_mtime:
                subprocess.run(
                    ["gcc", "-O3", "-fPIC", "-shared", str(c_path), "-o", str(so_path), "-lm"],
                    check=True, capture_output=True, text=True,
                )
            library = ctypes.CDLL(str(so_path))
            func = library.battery_symbolic_predict
            func.argtypes = [ctypes.POINTER(ctypes.c_double)]
            func.restype = ctypes.c_double
            self.symbolic_c = (library, func)
        except Exception as exc:
            print(f"Symbolic C compile/load failed; using JSON expression: {exc}")

    def predict_scaled(self, X):
        X = np.asarray(X, dtype=np.float32)
        fmt = self.model_format

        if fmt == "exact_gpr_numpy":
            d = self.data
            kind = str(d["kernel_kind"].item())
            ls = d["lengthscale"]
            outscale = float(d["outputscale"])
            if kind == "rbf":
                K = _rbf(X, d["X_train"], ls, outscale)
            elif kind == "matern":
                K = _matern(X, d["X_train"], ls, outscale, float(d["nu"]))
            else:
                raise ValueError(f"Unknown exact GPR kernel kind {kind!r}.")
            return K @ d["alpha"]

        if fmt == "rbf_svr_numpy":
            d = self.data
            support = d["support_vectors"]
            gamma = float(d["gamma"])
            dual = d["dual_coef"]
            intercept = float(d["intercept"])
            out = np.empty(len(X), dtype=np.float64)
            for start in range(0, len(X), 128):
                xb = X[start:start + 128].astype(np.float64)
                delta = xb[:, None, :] - support[None, :, :]
                sq = np.einsum("nmd,nmd->nm", delta, delta, optimize=True)
                out[start:start + len(xb)] = np.exp(-gamma * sq) @ dual + intercept
            return out

        if fmt == "random_forest_arrays":
            return _predict_rf_arrays(X, self.data)

        if fmt == "sparse_gpr_numpy":
            d = self.data
            return float(d["mean"]) + _rbf(
                X, d["inducing"], d["lengthscale"], float(d["outputscale"])
            ) @ d["weights"]

        if fmt == "deep_kernel_gpr_numpy_plus_onnx":
            d = self.data
            Z = self.feature_extractor.predict(X).astype(np.float64)
            return float(d["mean"]) + _rbf(
                Z, d["train_z"], d["lengthscale"], float(d["outputscale"])
            ) @ d["alpha"]

        if fmt == "onnx_neural":
            return self.onnx.predict(X)

        if fmt == "xgboost_ubjson":
            import xgboost as xgb
            return np.asarray(self.booster.predict(xgb.DMatrix(X)), dtype=np.float64).reshape(-1)

        if fmt == "symbolic_c_json":
            if self.symbolic_c is not None:
                _, func = self.symbolic_c
                ptr_type = ctypes.POINTER(ctypes.c_double)
                outputs = np.empty(len(X), dtype=np.float64)
                for i, row in enumerate(np.asarray(X, dtype=np.float64)):
                    outputs[i] = func(row.ctypes.data_as(ptr_type))
                return outputs
            return np.asarray(_eval_symbolic(self.symbolic_tree, X), dtype=np.float64).reshape(-1)

        raise ValueError(f"Unsupported model format {fmt!r}.")


class TegraStatsMonitor:
    """Integrate board VDD_IN power samples emitted by tegrastats."""
    PATTERN = re.compile(r"VDD_IN\s+([0-9]+(?:\.[0-9]+)?)mW", re.IGNORECASE)

    def __init__(self, interval_ms=100):
        self.interval_ms = int(interval_ms)
        self.samples = []
        self.proc = None
        self.thread = None
        self.stop_event = threading.Event()

    def start(self):
        command = shutil.which("tegrastats")
        if command is None:
            return False
        try:
            self.proc = subprocess.Popen(
                [command, "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()
            time.sleep(0.15)
            return self.proc.poll() is None
        except Exception:
            self.proc = None
            return False

    def _reader(self):
        while not self.stop_event.is_set() and self.proc and self.proc.stdout:
            line = self.proc.stdout.readline()
            if not line:
                break
            match = self.PATTERN.search(line)
            if match:
                self.samples.append((time.perf_counter(), float(match.group(1)) / 1000.0))

    def stop(self):
        self.stop_event.set()
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.thread:
            self.thread.join(timeout=1)
        if len(self.samples) < 2:
            return {"energy_J_batch_total": float("nan"), "mean_power_W": float("nan"),
                    "power_samples": len(self.samples), "power_source": "unavailable"}
        ts = np.asarray([x[0] for x in self.samples], dtype=np.float64)
        watts = np.asarray([x[1] for x in self.samples], dtype=np.float64)
        energy = float(np.trapezoid(watts, ts) if hasattr(np, "trapezoid") else np.trapz(watts, ts))
        return {
            "energy_J_batch_total": energy,
            "mean_power_W": float(np.mean(watts)),
            "power_samples": len(watts),
            "power_source": "tegrastats VDD_IN",
        }


def prepare_workload(frame: pd.DataFrame, metadata: dict, model_name: str, cycles_per_cell: int):
    X_all = transform_from_metadata(frame, metadata)
    seq_len = int(metadata.get("sequence_length") or SEQUENCE_LENGTH_DEFAULT)
    X_input = make_sequences(frame, X_all, seq_len) if model_name == "bilstm" else X_all

    frame = frame.copy()
    frame["_row_id"] = np.arange(len(frame))
    subset = (
        frame.groupby("cell_group", sort=True, group_keys=False)
        .head(int(cycles_per_cell))
        .copy()
        .reset_index(drop=True)
    )
    if subset.empty:
        raise ValueError("No cycle rows selected for the hardware workload.")
    positions = subset["_row_id"].to_numpy(dtype=int)
    return subset, X_input[positions]


def run_benchmark(model_name: str, script_path: str | Path, cycles_per_cell=8,
                  repeats=30, warmup=3, provider="auto", data_path=None):
    project_root = find_project_root(script_path)
    model_dir = project_root / "models" / "optimized" / model_name
    metadata_path = model_dir / "metadata.json"
    metadata = _safe_json_load(metadata_path)

    if metadata.get("model_name") != model_name:
        raise ValueError(f"Artifact metadata model_name={metadata.get('model_name')!r} does not match {model_name!r}.")
    if int(metadata.get("format_version", 0)) < 3:
        raise ValueError(
            f"Expected translator metadata format_version >= 3, found {metadata.get('format_version')}. "
            "Run translate_jetson_ready.py for this model first."
        )
    parquet_path = Path(data_path).expanduser().resolve() if data_path else project_root / "data" / "unified.parquet"
    frame = load_cycle_frame(parquet_path, metadata)
    subset, X = prepare_workload(frame, metadata, model_name, cycles_per_cell)

    predictor = OptimizedPredictor(model_name, model_dir, metadata, provider=provider)
    print(f"Model: {model_name}")
    print(f"Optimized format: {metadata['model_format']}")
    print(f"Runtime provider: {predictor.provider}")
    print(f"Parquet: {parquet_path}")
    print(f"Hardware workload: {len(subset)} cycles across {subset['cell_group'].nunique()} cells")
    print(f"Transformed workload shape: {X.shape}")
    print("Note: this is an inference-cost workload. Accuracy/generalization must be taken from LOCO OOF results.")

    for _ in range(max(0, int(warmup))):
        predictor.predict_scaled(X)

    monitor = TegraStatsMonitor()
    power_started = monitor.start() if is_jetson() else False
    times = []
    last_pred = None
    for _ in range(max(1, int(repeats))):
        started = time.perf_counter()
        last_pred = predictor.predict_scaled(X)
        times.append(time.perf_counter() - started)
    power = monitor.stop() if power_started else {
        "energy_J_batch_total": float("nan"), "mean_power_W": float("nan"),
        "power_samples": 0, "power_source": "not running on Jetson/tegrastats unavailable",
    }

    times_ms = np.asarray(times, dtype=np.float64) * 1000.0
    mean_batch_ms = float(times_ms.mean())
    p50_batch_ms = float(np.percentile(times_ms, 50))
    p95_batch_ms = float(np.percentile(times_ms, 95))
    samples_processed = int(len(X) * max(1, int(repeats)))
    energy_per_sample = (
        float(power["energy_J_batch_total"]) / samples_processed
        if np.isfinite(power["energy_J_batch_total"]) and samples_processed > 0 else float("nan")
    )
    latency_per_sample = mean_batch_ms / max(1, len(X))
    total_workload_energy = float(power["energy_J_batch_total"])

    prediction_frame = subset[["source_dataset", "entity_id", "cycle_id", "cycle_order", "cell_group"]].copy()
    target_mean = float(metadata["target_mean"])
    target_scale = float(metadata["target_scale"])
    prediction_frame["predicted_soh_pct"] = np.asarray(last_pred, dtype=np.float64) * target_scale + target_mean

    # Store only hardware/deployment results here; never overwrite LOCO accuracy results.
    results_dir = project_root / "baseline_results" / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(results_dir / "jetson_inference_predictions.csv", index=False)

    metrics = {
        "model": model_name,
        "model_format": metadata["model_format"],
        "benchmark_device": platform.platform(),
        "is_jetson": bool(is_jetson()),
        "runtime_provider": predictor.provider,
        "n_samples_per_batch": int(len(X)),
        "n_cells": int(subset["cell_group"].nunique()),
        "cycles_per_cell_cap": int(cycles_per_cell),
        "repeats": int(max(1, int(repeats))),
        "warmup": int(max(0, int(warmup))),
        "batch_latency_mean_ms": mean_batch_ms,
        "batch_latency_p50_ms": p50_batch_ms,
        "batch_latency_p95_ms": p95_batch_ms,
        "jetson_latency_ms": latency_per_sample if is_jetson() else float("nan"),
        "throughput_samples_per_second": (
            len(X) / (mean_batch_ms / 1000.0) if mean_batch_ms > 0 else float("nan")
        ),
        "jetson_energy_J": energy_per_sample if is_jetson() else float("nan"),
        "jetson_energy_J_batch_total": total_workload_energy if is_jetson() else float("nan"),
        "mean_power_W": power["mean_power_W"],
        "power_samples": int(power["power_samples"]),
        "power_source": power["power_source"],
        "artifact_size_MB": float(sum(p.stat().st_size for p in model_dir.rglob("*") if p.is_file()) / (1024**2)),
        "equation_node_count": metadata.get("equation_node_count", float("nan")),
        "target_mean": target_mean,
        "target_scale": target_scale,
        "measurement_scope": "optimized artifact inference only; use notebook LOCO OOF metrics for accuracy",
        "run_utc": datetime.now(timezone.utc).isoformat(),
    }

    with (results_dir / "jetson_hardware_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, allow_nan=True)
    pd.DataFrame([{"Metric": k, "Value": v} for k, v in metrics.items()]).to_csv(
        results_dir / "jetson_hardware_metrics.csv", index=False
    )

    print("\nHARDWARE METRICS")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"\nPredictions: {results_dir / 'jetson_inference_predictions.csv'}")
    print(f"Hardware metrics: {results_dir / 'jetson_hardware_metrics.json'}")
    if not is_jetson():
        print("Host smoke test only: jetson_latency_ms and jetson_energy_J are intentionally NaN.")
    elif not np.isfinite(energy_per_sample):
        print("Latency was measured on Jetson, but VDD_IN energy was unavailable (insufficient tegrastats samples).")
    return metrics


def main(model_name: str, script_path: str | Path):
    parser = argparse.ArgumentParser(
        description=f"Optimized Jetson inference benchmark for {model_name}."
    )
    parser.add_argument("--cycles-per-cell", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--provider", choices=["auto", "cpu", "cuda", "tensorrt"], default="auto")
    parser.add_argument("--data", type=str, default=None, help="Optional path to cycle-level unified.parquet.")
    args = parser.parse_args()
    if args.cycles_per_cell < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("cycles-per-cell/repeats must be positive and warmup nonnegative.")
    run_benchmark(
        model_name, script_path,
        cycles_per_cell=args.cycles_per_cell,
        repeats=args.repeats,
        warmup=args.warmup,
        provider=args.provider,
        data_path=args.data,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--script-path", default=__file__)
    parser.add_argument("--cycles-per-cell", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--provider", choices=["auto", "cpu", "cuda", "tensorrt"], default="auto")
    parser.add_argument("--data")
    args = parser.parse_args()
    if args.list_models:
        print("\n".join(MODEL_NAMES))
    elif args.model:
        run_benchmark(args.model, args.script_path, args.cycles_per_cell,
                      args.repeats, args.warmup, args.provider, args.data)
    else:
        parser.error("Use --list-models or --model MODEL_NAME.")
