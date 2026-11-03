#!/usr/bin/env python3
"""Inference and Jetson Orin Nano hardware-only benchmark for optimized exports.

Reads models/optimized/<model>/ (never the original .joblib files), benchmarks a
small deterministic chronological subset, and writes ONLY:
  equation_node_count, jetson_latency_ms, jetson_energy_J
into baseline_results/<model>/jetson_hardware_metrics.{csv,json}.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RANDOM_SEED = 42
TARGET = "soh_pct"
AGGREGATIONS = ["mean", "std", "min", "max", "first", "last"]
NUMERIC_RAW_FEATURES = [
    "voltage_V", "current_A", "current_avg_A", "temperature_C", "soc_pct",
    "charge_counter_Ah", "nominal_capacity_Ah", "elapsed_time_s",
    "impedance_re_ohm", "impedance_rct_ohm", "cycle_count", "plugged",
]
CATEGORICAL_RAW_FEATURES = ["source_dataset", "operation", "battery_technology"]


def find_project_root(script_path):
    p = Path(script_path).resolve()
    for candidate in [p.parent, *p.parents]:
        if (candidate / "data" / "unified.parquet").exists() and (candidate / "models" / "optimized").exists():
            return candidate
    raise FileNotFoundError(
        "Cannot locate project root. Expected data/unified.parquet and models/optimized/."
    )


def cycle_number(value):
    if pd.isna(value):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        found = re.findall(r"-?\d+(?:\.\d+)?", str(value))
        return float(found[-1]) if found else np.nan


def load_measurements(dataset_path):
    available = set(pq.ParquetFile(dataset_path).schema_arrow.names)
    requested = [
        "source_dataset", "entity_id", "session_id", "cycle_id", "operation",
        "timestamp_utc", "soh_pct", "measured_capacity_Ah", "capacity_raw_Ah",
        "nominal_capacity_Ah", *NUMERIC_RAW_FEATURES, "battery_technology",
    ]
    # Deduplicate while preserving column order.
    read_columns = list(dict.fromkeys(c for c in requested if c in available))
    try:
        frame = pd.read_parquet(
            dataset_path, columns=read_columns,
            filters=[("source_dataset", "in", ["NASA", "CALCE"])],
        )
    except Exception:
        frame = pd.read_parquet(dataset_path, columns=read_columns)
    if "source_dataset" not in frame or "entity_id" not in frame:
        raise ValueError("unified.parquet must contain source_dataset and entity_id.")
    frame["source_dataset"] = frame["source_dataset"].astype(str).str.upper()
    frame = frame[frame["source_dataset"].isin(["NASA", "CALCE"])].copy()
    if frame.empty:
        raise ValueError("No NASA/CALCE records found in unified.parquet.")
    return frame


def build_cycle_table(measurements):
    """Reproduce the notebooks' labelled cycle aggregation for benchmark input."""
    df = measurements.copy()
    for col in ["session_id", "cycle_id", "operation", "timestamp_utc"]:
        if col not in df:
            df[col] = pd.NA
    for col in ["source_dataset", "entity_id"]:
        df[col] = df[col].astype("string").fillna("UNKNOWN")
    df["session_id"] = df["session_id"].astype("string").fillna("")
    df["cycle_id"] = df["cycle_id"].astype("string").replace(
        {"<NA>": pd.NA, "": pd.NA, "nan": pd.NA}
    )
    if TARGET not in df:
        df[TARGET] = np.nan
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    if {"measured_capacity_Ah", "nominal_capacity_Ah"}.issubset(df.columns):
        measured = pd.to_numeric(df["measured_capacity_Ah"], errors="coerce")
        nominal = pd.to_numeric(df["nominal_capacity_Ah"], errors="coerce")
        derived = 100.0 * measured / nominal.where(nominal > 0)
        df[TARGET] = df[TARGET].where(df[TARGET].notna(), derived)
    if {"capacity_raw_Ah", "nominal_capacity_Ah"}.issubset(df.columns):
        raw = pd.to_numeric(df["capacity_raw_Ah"], errors="coerce")
        nominal = pd.to_numeric(df["nominal_capacity_Ah"], errors="coerce")
        derived = 100.0 * raw / nominal.where(nominal > 0)
        df[TARGET] = df[TARGET].where(df[TARGET].notna(), derived)
    df[TARGET] = df[TARGET].replace([np.inf, -np.inf], np.nan)

    cycle_key = df["cycle_id"].where(df["cycle_id"].notna(), df["session_id"])
    cycle_key = cycle_key.replace("", pd.NA)
    if "cycle_count" in df:
        cc = pd.to_numeric(df["cycle_count"], errors="coerce").astype("string")
        cycle_key = cycle_key.where(cycle_key.notna(), cc)
    df["_cycle_key"] = cycle_key.fillna("NO_CYCLE_ID").astype("string")

    group_cols = ["source_dataset", "entity_id", "session_id", "_cycle_key"]
    labelled = df.loc[df[TARGET].notna(), group_cols].drop_duplicates()
    if labelled.empty:
        raise ValueError(
            "No measured SOH/capacity labels found. This benchmark subset is drawn "
            "from labelled NASA/CALCE cycles; verify unified.parquet labels."
        )
    keys = pd.MultiIndex.from_frame(labelled)
    all_keys = pd.MultiIndex.from_frame(df[group_cols])
    df = df.loc[all_keys.isin(keys)].copy()

    raw_numeric = [c for c in NUMERIC_RAW_FEATURES if c in df.columns]
    raw_categorical = [c for c in CATEGORICAL_RAW_FEATURES if c in df.columns]
    rows = []
    for key, group in df.groupby(group_cols, sort=False, dropna=False):
        source, entity, session, cycle = key
        if "timestamp_utc" in group:
            group = group.sort_values("timestamp_utc", kind="mergesort", na_position="last")
        labels = pd.to_numeric(group[TARGET], errors="coerce").dropna()
        if labels.empty:
            continue
        row = {
            "source_dataset": str(source), "entity_id": str(entity),
            "session_id": str(session), "cycle_key": str(cycle),
            TARGET: float(labels.median()), "n_measurements": int(len(group)),
            "cycle_index": cycle_number(cycle), "cycle_timestamp": pd.NaT,
        }
        for col in raw_categorical:
            vals = group[col].dropna().astype(str)
            row[col] = vals.mode().iloc[0] if not vals.empty else "UNKNOWN"
        for col in raw_numeric:
            vals = pd.to_numeric(group[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if vals.notna().any():
                stats = {
                    "mean": vals.mean(), "std": vals.std(), "min": vals.min(),
                    "max": vals.max(), "first": vals.iloc[0], "last": vals.iloc[-1],
                }
            else:
                stats = {stat: np.nan for stat in AGGREGATIONS}
            for stat in AGGREGATIONS:
                row[f"{col}__{stat}"] = stats[stat]
        if "timestamp_utc" in group:
            ts = pd.to_datetime(group["timestamp_utc"], errors="coerce", utc=True).dropna()
            row["cycle_timestamp"] = ts.min() if not ts.empty else pd.NaT
        rows.append(row)

    cycles = pd.DataFrame(rows)
    if cycles.empty:
        raise ValueError("No labelled cycles could be aggregated.")
    cycles["cycle_index"] = pd.to_numeric(cycles["cycle_index"], errors="coerce")
    cycles = cycles.sort_values(
        ["source_dataset", "entity_id", "cycle_index", "cycle_timestamp", "cycle_key"],
        kind="mergesort", na_position="last",
    ).reset_index(drop=True)
    missing = cycles["cycle_index"].isna()
    if missing.any():
        cycles.loc[missing, "cycle_index"] = (
            cycles.loc[missing].groupby(["source_dataset", "entity_id"], sort=False)
            .cumcount().astype(float)
        )
    cycles["cell_group"] = cycles["source_dataset"].astype(str) + "::" + cycles["entity_id"].astype(str)
    return cycles.replace([np.inf, -np.inf], np.nan)


def transform_from_metadata(frame, meta):
    """Apply exported train-fitted imputation, scaling and one-hot parameters."""
    numeric = meta.get("numeric_features", [])
    categorical = meta.get("categorical_features", [])
    blocks = []

    num_cfg = meta.get("numeric_preprocessing")
    if numeric:
        raw = np.empty((len(frame), len(numeric)), dtype=np.float64)
        for j, col in enumerate(numeric):
            if col in frame:
                raw[:, j] = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
            else:
                raw[:, j] = np.nan
        missing = ~np.isfinite(raw)
        stats = np.asarray(num_cfg["imputer_statistics"], dtype=np.float64)
        imputed = np.where(missing, stats[None, :], raw)
        indicator_features = np.asarray(num_cfg.get("indicator_features", []), dtype=int)
        if indicator_features.size:
            indicators = missing[:, indicator_features].astype(np.float64)
            num_matrix = np.concatenate([imputed, indicators], axis=1)
        else:
            num_matrix = imputed
        mean = np.asarray(num_cfg["scaler_mean"], dtype=np.float64)
        scale = np.asarray(num_cfg["scaler_scale"], dtype=np.float64)
        scale = np.where(scale == 0, 1.0, scale)
        blocks.append(((num_matrix - mean[None, :]) / scale[None, :]).astype(np.float32))

    cat_cfg = meta.get("categorical_preprocessing")
    if categorical:
        cat_blocks = []
        modes = cat_cfg["imputer_statistics"]
        categories = cat_cfg["categories"]
        for j, col in enumerate(categorical):
            if col in frame:
                raw_values = frame[col].to_numpy(dtype=object)
                known_categories = set(str(v) for v in categories[j])
                normalized = []
                for value in raw_values:
                    # sklearn's object-dtype OneHotEncoder may preserve Python
                    # None as a category (rather than treating it as np.nan).
                    if value is None and "None" in known_categories:
                        normalized.append("None")
                    elif pd.isna(value):
                        normalized.append(str(modes[j]))
                    else:
                        normalized.append(str(value))
                values = np.asarray(normalized, dtype=object)
            else:
                values = np.asarray([str(modes[j])] * len(frame), dtype=object)
            cat_blocks.append(np.column_stack([
                (values == str(category)).astype(np.float32)
                for category in categories[j]
            ]) if categories[j] else np.zeros((len(frame), 0), dtype=np.float32))
        if cat_blocks:
            blocks.append(np.concatenate(cat_blocks, axis=1).astype(np.float32))

    if not blocks:
        raise ValueError("No preprocessing features were exported in metadata.json.")
    X = np.concatenate(blocks, axis=1).astype(np.float32)
    expected = len(meta["transformed_feature_names"])
    if X.shape[1] != expected:
        raise ValueError(
            f"Preprocessing dimension mismatch: generated {X.shape[1]} features, expected {expected}."
        )
    return X


def make_sequences(table, X, sequence_length):
    X = np.asarray(X, dtype=np.float32)
    out = np.zeros((len(table), sequence_length, X.shape[1]), dtype=np.float32)
    for _, indices in table.groupby("cell_group", sort=False).groups.items():
        pos = np.asarray(list(indices), dtype=int)
        order = np.argsort(table.iloc[pos]["cycle_index"].to_numpy(), kind="stable")
        pos = pos[order]
        for j, end_pos in enumerate(pos):
            start = max(0, j - sequence_length + 1)
            win = pos[start:j + 1]
            out[end_pos, -len(win):, :] = X[win]
    return out


def _rbf(X, Z, lengthscale, outputscale=1.0, chunk=64):
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    ls = np.asarray(lengthscale, dtype=np.float64).reshape(-1)
    if ls.size == 1:
        ls = np.repeat(ls, X.shape[1])
    result = np.empty((len(X), len(Z)), dtype=np.float64)
    Zs = Z / ls[None, :]
    for start in range(0, len(X), chunk):
        Xs = X[start:start + chunk] / ls[None, :]
        x2 = np.einsum("ij,ij->i", Xs, Xs)[:, None]
        z2 = np.einsum("ij,ij->i", Zs, Zs)[None, :]
        d2 = np.maximum(x2 + z2 - 2.0 * (Xs @ Zs.T), 0.0)
        result[start:start + chunk] = float(outputscale) * np.exp(-0.5 * d2)
    return result


def _matern(X, Z, lengthscale, outputscale, nu, chunk=64):
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    ls = np.asarray(lengthscale, dtype=np.float64).reshape(-1)
    if ls.size == 1:
        ls = np.repeat(ls, X.shape[1])
    result = np.empty((len(X), len(Z)), dtype=np.float64)
    Zs = Z / ls[None, :]
    for start in range(0, len(X), chunk):
        Xs = X[start:start + chunk] / ls[None, :]
        x2 = np.einsum("ij,ij->i", Xs, Xs)[:, None]
        z2 = np.einsum("ij,ij->i", Zs, Zs)[None, :]
        r = np.sqrt(np.maximum(x2 + z2 - 2.0 * (Xs @ Zs.T), 0.0))
        if abs(nu - 0.5) < 1e-8:
            k = np.exp(-r)
        elif abs(nu - 1.5) < 1e-8:
            q = math.sqrt(3.0) * r
            k = (1.0 + q) * np.exp(-q)
        elif abs(nu - 2.5) < 1e-8:
            q = math.sqrt(5.0) * r
            k = (1.0 + q + q * q / 3.0) * np.exp(-q)
        else:
            raise ValueError(f"Unsupported Matérn nu={nu}.")
        result[start:start + chunk] = float(outputscale) * k
    return result


def eval_symbolic_tree(node, X):
    if "feature" in node:
        return X[:, int(node["feature"])]
    if "constant" in node:
        return np.full(len(X), float(node["constant"]), dtype=np.float64)
    args = [eval_symbolic_tree(a, X) for a in node["args"]]
    op = node["op"]
    if op == "add": return args[0] + args[1]
    if op == "sub": return args[0] - args[1]
    if op == "mul": return args[0] * args[1]
    if op == "div": return np.where(np.abs(args[1]) > 0.001, args[0] / np.where(np.abs(args[1]) > 0.001, args[1], 1.0), 1.0)
    if op == "sqrt": return np.sqrt(np.abs(args[0]))
    if op == "log": return np.where(np.abs(args[0]) > 0.001, np.log(np.maximum(np.abs(args[0]), 0.001)), 0.0)
    if op == "abs": return np.abs(args[0])
    if op == "neg": return -args[0]
    raise ValueError(f"Unsupported symbolic op {op!r}.")


class TensorRTRunner:
    """Minimal single-input/single-output TensorRT runner (TRT 8.x/10.x)."""
    def __init__(self, engine_path):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401 - creates a CUDA context
        self.trt, self.cuda = trt, cuda
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"TensorRT could not deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self._buffer_shape = None
        self._host_buffers = None
        self._device_buffers = None
        self._legacy_bindings = None
        self.input_name = None
        self.output_name = None
        if hasattr(self.engine, "num_io_tensors"):
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT: self.input_name = name
                else: self.output_name = name
            if self.input_name is None or self.output_name is None:
                raise RuntimeError("Expected one input and one output TensorRT tensor.")
            self.new_api = True
        else:
            self.new_api = False
            self.input_idx = next(i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i))
            self.output_idx = next(i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i))

    def predict(self, X):
        X = np.ascontiguousarray(X)
        shape_key = tuple(X.shape)
        if self._buffer_shape != shape_key:
            self._allocate_buffers(X)
        host_buffers = self._host_buffers
        device_buffers = self._device_buffers

        if self.new_api:
            input_host = host_buffers[self.input_name]
            np.copyto(input_host, X.astype(input_host.dtype, copy=False))
            self.cuda.memcpy_htod_async(device_buffers[self.input_name], input_host, self.stream)
            if not self.context.execute_async_v3(self.stream.handle):
                raise RuntimeError("TensorRT execute_async_v3 failed.")
            self.cuda.memcpy_dtoh_async(host_buffers[self.output_name], device_buffers[self.output_name], self.stream)
            self.stream.synchronize()
            return np.asarray(host_buffers[self.output_name]).reshape(len(X), -1).squeeze(-1)

        input_host = host_buffers[self.input_idx]
        np.copyto(input_host, X.astype(input_host.dtype, copy=False))
        self.cuda.memcpy_htod_async(device_buffers[self.input_idx], input_host, self.stream)
        if not self.context.execute_async_v2(self._legacy_bindings, self.stream.handle):
            raise RuntimeError("TensorRT execute_async_v2 failed.")
        self.cuda.memcpy_dtoh_async(host_buffers[self.output_idx], device_buffers[self.output_idx], self.stream)
        self.stream.synchronize()
        return np.asarray(host_buffers[self.output_idx]).reshape(len(X), -1).squeeze(-1)

    def _allocate_buffers(self, X):
        if self.new_api:
            self.context.set_input_shape(self.input_name, tuple(X.shape))
            names = [self.input_name, self.output_name]
            hosts, devices = {}, {}
            for name in names:
                shape = tuple(self.context.get_tensor_shape(name))
                if any(int(d) < 0 for d in shape):
                    raise RuntimeError(f"TensorRT could not resolve output/input shape for {name}: {shape}")
                dtype = self.trt.nptype(self.engine.get_tensor_dtype(name))
                host = self.cuda.pagelocked_empty(int(np.prod(shape)), dtype).reshape(shape)
                device = self.cuda.mem_alloc(host.nbytes)
                hosts[name], devices[name] = host, device
                self.context.set_tensor_address(name, int(device))
            self._host_buffers, self._device_buffers = hosts, devices
        else:
            self.context.set_binding_shape(self.input_idx, tuple(X.shape))
            bindings = [None] * self.engine.num_bindings
            hosts, devices = {}, {}
            for i in range(self.engine.num_bindings):
                shape = tuple(self.context.get_binding_shape(i))
                if any(int(d) < 0 for d in shape):
                    raise RuntimeError(f"TensorRT could not resolve binding shape {i}: {shape}")
                dtype = self.trt.nptype(self.engine.get_binding_dtype(i))
                host = self.cuda.pagelocked_empty(int(np.prod(shape)), dtype).reshape(shape)
                device = self.cuda.mem_alloc(host.nbytes)
                bindings[i] = int(device)
                hosts[i], devices[i] = host, device
            self._host_buffers, self._device_buffers = hosts, devices
            self._legacy_bindings = bindings
        self._buffer_shape = tuple(X.shape)


class OnnxRunner:
    def __init__(self, onnx_path, engine_path, provider="auto", cache_dir=None):
        self.runner = None
        self.kind = None
        if provider in ("auto", "tensorrt") and Path(engine_path).exists():
            try:
                self.runner = TensorRTRunner(engine_path)
                self.kind = "TensorRT engine"
                return
            except Exception as exc:
                if provider == "tensorrt":
                    raise RuntimeError(f"Could not load TensorRT engine: {exc}") from exc
                print(f"TensorRT engine unavailable; trying ONNX Runtime ({exc})")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "ONNX inference needs ONNX Runtime, or a built TensorRT engine plus "
                "TensorRT/PyCUDA. Install a JetPack-compatible ONNX Runtime build."
            ) from exc
        available = ort.get_available_providers()
        providers = []
        if provider in ("auto", "tensorrt") and "TensorrtExecutionProvider" in available:
            options = {
                "trt_fp16_enable": "True",
                "trt_engine_cache_enable": "True",
                "trt_engine_cache_path": str(cache_dir or Path(onnx_path).parent / "trt_cache"),
            }
            providers.append(("TensorrtExecutionProvider", options))
        elif provider == "tensorrt":
            raise RuntimeError("TensorRT provider is unavailable and no .engine file was found.")
        if provider in ("auto", "cuda") and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        if provider == "cuda" and "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider is not available in this ONNX Runtime build.")
        providers.append("CPUExecutionProvider")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        self.runner = ort.InferenceSession(str(onnx_path), sess_options=options, providers=providers)
        self.input_name = self.runner.get_inputs()[0].name
        self.output_name = self.runner.get_outputs()[0].name
        self.kind = ",".join(self.runner.get_providers())

    def predict(self, X):
        if isinstance(self.runner, TensorRTRunner):
            return self.runner.predict(X)
        output = self.runner.run([self.output_name], {self.input_name: np.asarray(X, dtype=np.float32)})[0]
        return np.asarray(output).reshape(len(X), -1).squeeze(-1)


class OptimizedPredictor:
    def __init__(self, model_name, model_dir, metadata, provider="auto"):
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.meta = metadata
        self.target_mean = float(metadata["target_mean"])
        self.target_scale = float(metadata["target_scale"])
        self.format = metadata["model_format"]
        self.model_data = None
        self.onnx = None
        self.feature_onnx = None
        self.booster = None
        self.c_function = None

        if self.format in ("exact_gpr_numpy", "rbf_svr_numpy", "random_forest_arrays", "sparse_gpr_numpy"):
            self.model_data = dict(np.load(self.model_dir / "model.npz", allow_pickle=False))
        elif self.format == "deep_kernel_gpr_numpy_plus_onnx":
            self.model_data = dict(np.load(self.model_dir / "gp_params.npz", allow_pickle=False))
            self.feature_onnx = OnnxRunner(
                self.model_dir / metadata["onnx_file"], self.model_dir / "feature_extractor.engine",
                provider, self.model_dir / "trt_cache",
            )
        elif self.format == "onnx_neural":
            self.onnx = OnnxRunner(
                self.model_dir / metadata["onnx_file"], self.model_dir / "model.engine",
                provider, self.model_dir / "trt_cache",
            )
        elif self.format == "xgboost_ubjson":
            try:
                import xgboost as xgb
            except ImportError as exc:
                raise ImportError("This model needs xgboost installed on the Jetson.") from exc
            self.booster = xgb.Booster()
            self.booster.load_model(str(self.model_dir / "model.ubj"))
        elif self.format == "symbolic_c_json":
            self.symbolic_tree = json.loads((self.model_dir / metadata["expression_file"]).read_text())
            self._load_symbolic_c()
        else:
            raise ValueError(f"Unsupported optimized model format: {self.format}")

    def _load_symbolic_c(self):
        c_path = self.model_dir / "equation.c"
        so_path = self.model_dir / "equation.so"
        compiler = shutil.which("gcc") or shutil.which("cc")
        if compiler and c_path.exists():
            if (not so_path.exists()) or so_path.stat().st_mtime < c_path.stat().st_mtime:
                command = [compiler, "-O3", "-fPIC", "-shared", str(c_path), "-o", str(so_path), "-lm"]
                subprocess.run(command, check=True, capture_output=True, text=True)
        if so_path.exists():
            lib = ctypes.CDLL(str(so_path))
            fn = lib.battery_symbolic_predict
            fn.argtypes = [ctypes.POINTER(ctypes.c_double)]
            fn.restype = ctypes.c_double
            self.c_function = fn

    def _symbolic_predict(self, X):
        if self.c_function is None:
            return eval_symbolic_tree(self.symbolic_tree, np.asarray(X, dtype=np.float64))
        X = np.ascontiguousarray(X, dtype=np.float64)
        out = np.empty(len(X), dtype=np.float64)
        ptr_type = ctypes.POINTER(ctypes.c_double)
        for i in range(len(X)):
            out[i] = self.c_function(X[i].ctypes.data_as(ptr_type))
        return out

    def predict_scaled(self, X):
        X = np.asarray(X, dtype=np.float32)
        d = self.model_data
        if self.format == "exact_gpr_numpy":
            kind = str(d["kernel_kind"].item())
            if kind == "rbf":
                K = _rbf(X, d["X_train"], d["lengthscale"], float(d["outputscale"]))
            else:
                K = _matern(X, d["X_train"], d["lengthscale"], float(d["outputscale"]), float(d["nu"]))
            return K @ d["alpha"]
        if self.format == "rbf_svr_numpy":
            support = d["support_vectors"]
            gamma = float(d["gamma"])
            coef = d["dual_coef"]
            out = np.empty(len(X), dtype=np.float64)
            for start in range(0, len(X), 64):
                block = X[start:start + 64].astype(np.float64)
                d2 = np.maximum(
                    np.einsum("ij,ij->i", block, block)[:, None]
                    + np.einsum("ij,ij->i", support, support)[None, :]
                    - 2 * block @ support.T,
                    0.0,
                )
                out[start:start + len(block)] = np.exp(-gamma * d2) @ coef + float(d["intercept"])
            return out
        if self.format == "random_forest_arrays":
            left, right = d["left"], d["right"]
            features, thresholds, values = d["feature"], d["threshold"], d["value"]
            tree_preds = np.zeros((left.shape[0], len(X)), dtype=np.float64)
            for t in range(left.shape[0]):
                nodes = np.zeros(len(X), dtype=np.int32)
                for _ in range(int(d["node_count"][t])):
                    f = features[t, nodes]
                    active = f >= 0
                    if not np.any(active):
                        break
                    rows = np.flatnonzero(active)
                    feature_ids = f[active]
                    go_left = X[rows, feature_ids] <= thresholds[t, nodes[active]]
                    nodes[rows] = np.where(go_left, left[t, nodes[active]], right[t, nodes[active]])
                tree_preds[t] = values[t, nodes]
            return tree_preds.mean(axis=0)
        if self.format == "sparse_gpr_numpy":
            return float(d["mean"]) + _rbf(
                X, d["inducing"], d["lengthscale"], float(d["outputscale"])
            ) @ d["weights"]
        if self.format == "deep_kernel_gpr_numpy_plus_onnx":
            Z = self.feature_onnx.predict(X)
            return float(d["mean"]) + _rbf(
                Z, d["train_z"], d["lengthscale"], float(d["outputscale"])
            ) @ d["alpha"]
        if self.format == "onnx_neural":
            return self.onnx.predict(X)
        if self.format == "xgboost_ubjson":
            return np.asarray(self.booster.inplace_predict(X), dtype=np.float64).reshape(-1)
        if self.format == "symbolic_c_json":
            return self._symbolic_predict(X)
        raise RuntimeError(f"No predictor for format {self.format}")

    def predict(self, X):
        scaled = np.asarray(self.predict_scaled(X), dtype=np.float64).reshape(-1)
        return scaled * self.target_scale + self.target_mean


def is_jetson_device():
    try:
        model = Path("/proc/device-tree/model").read_bytes().replace(b"\x00", b" ").decode(errors="ignore").lower()
    except Exception:
        model = ""
    return ("jetson" in model and "nvidia" in model) or shutil.which("tegrastats") is not None


class TegraStatsMonitor:
    def __init__(self, interval_ms=100):
        self.interval_ms = int(interval_ms)
        self.proc = None
        self.thread = None
        self.samples = []
        self.stop_event = threading.Event()

    def start(self):
        executable = shutil.which("tegrastats")
        if not executable:
            return False
        try:
            self.proc = subprocess.Popen(
                [executable, "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1,
            )
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()
            return True
        except Exception:
            self.proc = None
            return False

    def _reader(self):
        pattern = re.compile(r"VDD_IN\s+(\d+)mW")
        if self.proc is None or self.proc.stdout is None:
            return
        while not self.stop_event.is_set():
            line = self.proc.stdout.readline()
            if not line:
                break
            match = pattern.search(line)
            if match:
                self.samples.append((time.perf_counter(), float(match.group(1)) / 1000.0))

    def stop(self):
        self.stop_event.set()
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.thread is not None:
            self.thread.join(timeout=2)
        if len(self.samples) < 2:
            return float("nan"), len(self.samples)
        times = np.asarray([x[0] for x in self.samples], dtype=float)
        watts = np.asarray([x[1] for x in self.samples], dtype=float)
        return float(np.trapezoid(watts, times)), len(self.samples)


def save_jetson_only_metrics(results_dir, metrics):
    results_dir.mkdir(parents=True, exist_ok=True)
    columns = ["equation_node_count", "jetson_latency_ms", "jetson_energy_J"]
    row = {key: metrics[key] for key in columns}
    csv_path = results_dir / "jetson_hardware_metrics.csv"
    json_path = results_dir / "jetson_hardware_metrics.json"
    csv_tmp = csv_path.with_suffix(".csv.tmp")
    json_tmp = json_path.with_suffix(".json.tmp")
    pd.DataFrame([row], columns=columns).to_csv(csv_tmp, index=False)
    json_tmp.write_text(json.dumps(row, indent=2, allow_nan=True), encoding="utf-8")
    os.replace(csv_tmp, csv_path)
    os.replace(json_tmp, json_path)
    return csv_path, json_path


def run_benchmark(model_name, script_path, cycles_per_cell=8, repeats=100,
                  warmup=5, provider="auto", tegrastats_interval_ms=100,
                  min_power_window_s=2.5):
    if cycles_per_cell < 1 or repeats < 1 or warmup < 0:
        raise ValueError("cycles-per-cell/repeats must be >=1 and warmup >=0.")
    root = find_project_root(script_path)
    data_path = root / "data" / "unified.parquet"
    model_dir = root / "models" / "optimized" / model_name
    results_dir = root / "baseline_results" / model_name
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Optimized export not found: {metadata_path}. Run python models/translate.py first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    predictor = OptimizedPredictor(model_name, model_dir, metadata, provider=provider)

    print("Reading benchmark subset from:", data_path)
    measurements = load_measurements(data_path)
    cycles = build_cycle_table(measurements)
    cycles = cycles.sort_values(
        ["cell_group", "cycle_index", "cycle_timestamp"],
        kind="mergesort", na_position="last",
    ).reset_index(drop=True)
    subset = (
        cycles.groupby("cell_group", sort=True, group_keys=False)
        .head(cycles_per_cell).copy().reset_index(drop=True)
    )
    if subset.empty:
        raise ValueError("The benchmark subset is empty.")

    # Metadata feature names are authoritative. Missing columns are imputed
    # according to the training-fitted preprocessing parameters.
    for col in metadata["feature_columns"]:
        if col not in subset:
            subset[col] = "UNKNOWN" if col in metadata["categorical_features"] else np.nan
    X_tabular = transform_from_metadata(subset, metadata)
    if model_name == "bilstm":
        X = make_sequences(subset, X_tabular, int(metadata.get("sequence_length", 8)))
    elif model_name == "deep_kernel_gpr":
        X = X_tabular
    else:
        X = X_tabular

    print(f"Model: {model_name}")
    print(f"Optimized format: {metadata['model_format']}")
    print(f"Subset: {len(subset)} cycles across {subset.cell_group.nunique()} cells")
    if predictor.onnx:
        print("Inference backend:", predictor.onnx.kind)
    if predictor.feature_onnx:
        print("Feature extractor backend:", predictor.feature_onnx.kind)
    print("Input shape:", tuple(X.shape))

    for _ in range(warmup):
        _ = predictor.predict(X)

    equation_count = metadata.get("equation_node_count", float("nan"))
    monitor = TegraStatsMonitor(tegrastats_interval_ms)
    monitor_started = monitor.start()
    times = []
    measurement_start = time.perf_counter()
    try:
        while len(times) < repeats or (
            monitor_started and time.perf_counter() - measurement_start < min_power_window_s
        ):
            start = time.perf_counter()
            prediction = np.asarray(predictor.predict(X)).reshape(-1)
            elapsed = (time.perf_counter() - start) * 1000.0
            if len(prediction) != len(X):
                raise RuntimeError(f"Got {len(prediction)} predictions for {len(X)} samples.")
            if not np.all(np.isfinite(prediction)):
                raise RuntimeError("Optimized model returned NaN/Inf predictions.")
            times.append(elapsed)
    finally:
        energy_j, power_samples = monitor.stop() if monitor_started else (float("nan"), 0)

    n = max(len(X), 1)
    latency_ms = float(np.mean(times) / n)
    energy_per_cycle = (
        float(energy_j / (len(times) * n))
        if monitor_started and np.isfinite(energy_j) and len(times) else float("nan")
    )
    metrics = {
        "equation_node_count": int(equation_count) if np.isfinite(equation_count) else float("nan"),
        "jetson_latency_ms": latency_ms,
        "jetson_energy_J": energy_per_cycle,
    }
    print("\nJetson-specific measurements:")
    print(pd.DataFrame([metrics]).to_string(index=False))

    if is_jetson_device() and monitor_started:
        csv_path, json_path = save_jetson_only_metrics(results_dir, metrics)
        print("Saved:", csv_path)
        print("Saved:", json_path)
        if not np.isfinite(energy_per_cycle):
            print(f"WARNING: only {power_samples} VDD_IN samples; energy unavailable. Increase --min-power-window-s.")
    else:
        print("No result file written: not detected as Jetson or tegrastats unavailable.")
        print("Host latency above is only a laptop compatibility measurement.")
    print("Existing LOCO metrics, OOF predictions, plots and XAI files were not changed.")
    return metrics


def main(model_name, script_path):
    parser = argparse.ArgumentParser(description=f"Jetson hardware-only benchmark for {model_name}.")
    parser.add_argument("--cycles-per-cell", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--provider", choices=["auto", "cpu", "cuda", "tensorrt"], default="auto",
                        help="ONNX backend preference. Auto prefers a TensorRT engine, then ONNX Runtime GPU, then CPU.")
    parser.add_argument("--tegrastats-interval-ms", type=int, default=100)
    parser.add_argument("--min-power-window-s", type=float, default=2.5)
    args = parser.parse_args()
    run_benchmark(
        model_name, script_path,
        cycles_per_cell=args.cycles_per_cell,
        repeats=args.repeats,
        warmup=args.warmup,
        provider=args.provider,
        tegrastats_interval_ms=args.tegrastats_interval_ms,
        min_power_window_s=args.min_power_window_s,
    )
