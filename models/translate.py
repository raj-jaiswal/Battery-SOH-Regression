#!/usr/bin/env python3
"""Export the trained battery baseline joblib bundles to Jetson-friendly formats.

Original .joblib files are never modified. Exports go to:
    models/optimized/<model_name>/

Formats:
  * Exact GPR / SVR / sparse GPR: NumPy arrays + lightweight inference kernels
  * Deep-kernel GPR: ONNX feature extractor + NumPy exact-GP parameters
  * Random Forest: compact tree arrays + NumPy tree evaluator
  * XGBoost: native UBJSON Booster
  * WideNN / DeepNN / Wide&DeepNN / BiLSTM: ONNX
  * Symbolic regression: expression JSON + portable C source

TensorRT engines are deliberately NOT built here. Build them on the Orin Nano
for its installed JetPack/TensorRT version using jetson/build_tensorrt_engines.sh.
"""
from __future__ import annotations

import argparse
import json
import math
import numbers
import os
import sys
import traceback
from pathlib import Path
from scipy.optimize import minimize

import joblib
import numpy as np

try:
    import torch
    from torch import nn
except Exception:
    torch = None
    nn = None

try:
    import gpytorch
except Exception:
    gpytorch = None

ROOT = Path(__file__).resolve().parent
OPT_ROOT = ROOT / "optimized"
MODEL_NAMES = [
    "gpr_rbf", "gpr_matern", "gpr_ard", "sparse_gpr",
    "deep_kernel_gpr", "svr", "random_forest", "xgboost",
    "wide_nn", "deep_nn", "wide_deep_nn", "bilstm",
    "symbolic_regression",
]

RANDOM_SEED = 42

def fast_ard_optimizer(obj_func, initial_theta, bounds):
    """
    Reimplementation of the optimizer used when training gpr_ard.
    Must remain at module level for joblib/pickle compatibility.
    """

    result = minimize(
        obj_func,
        initial_theta,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": 20,
            "maxfun": 30,
            "ftol": 1e-4,
            "maxls": 10,
        },
    )

    if not np.isfinite(result.fun):
        initial_loss = obj_func(
            initial_theta,
            eval_gradient=False
        )
        return initial_theta, float(initial_loss)

    return result.x, float(result.fun)

import __main__

__main__.fast_ard_optimizer = fast_ard_optimizer

def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

if torch is not None:
    class _FeedForwardNet(nn.Module):
        def __init__(self, d, kind, dropout=0.15):
            super().__init__()
            if kind == "wide":
                self.wide = nn.Linear(d, 1)
                self.deep = nn.Sequential(
                    nn.Linear(d, 64), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
                )
            elif kind == "deep":
                self.net = nn.Sequential(
                    nn.Linear(d, 128), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(32, 1),
                )
            elif kind == "wide_deep":
                self.wide = nn.Linear(d, 1)
                self.deep = nn.Sequential(
                    nn.Linear(d, 128), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
                )
            else:
                raise ValueError(kind)
            self.kind = kind

        def forward(self, x):
            if self.kind == "deep":
                return self.net(x)
            return self.wide(x) + self.deep(x)

    class _BiLSTMNet(nn.Module):
        def __init__(self, d, hidden=48, layers=2, dropout=0.15):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=d, hidden_size=hidden, num_layers=layers,
                batch_first=True, bidirectional=True,
                dropout=dropout if layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden * 2, hidden), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden, 1),
            )

        def forward(self, x):
            seq, _ = self.lstm(x)
            return self.head(seq[:, -1, :])

    class TorchRegressor:
        def __init__(self, kind="deep", epochs=120, batch_size=64, lr=1e-3,
                     weight_decay=1e-4, seed=RANDOM_SEED):
            self.kind, self.epochs, self.batch_size = kind, epochs, batch_size
            self.lr, self.weight_decay, self.seed = lr, weight_decay, seed

    class BiLSTMRegressor:
        def __init__(self, epochs=100, batch_size=64, lr=1e-3,
                     weight_decay=1e-4, seed=RANDOM_SEED):
            self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
            self.weight_decay, self.seed = weight_decay, seed
else:
    # Placeholders permit unrelated classical joblibs to be exported even when
    # torch is absent. Neural model translation will report a dependency error.
    class _FeedForwardNet: pass
    class _BiLSTMNet: pass
    class TorchRegressor: pass
    class BiLSTMRegressor: pass

if gpytorch is not None and torch is not None:
    class _SparseGPModel(gpytorch.models.ApproximateGP):
        def __init__(self, inducing_points):
            q = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
            strategy = gpytorch.variational.VariationalStrategy(
                self, inducing_points, q, learn_inducing_locations=True
            )
            super().__init__(strategy)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.size(-1))
            )

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

    class SparseGPRRegressor:
        def __init__(self, inducing_points=64, epochs=80, batch_size=128,
                     lr=0.01, seed=RANDOM_SEED):
            self.inducing_points, self.epochs = inducing_points, epochs
            self.batch_size, self.lr, self.seed = batch_size, lr, seed

        def predict_distribution(self, X):
            X = np.asarray(X, dtype=np.float32)
            self.model_.eval()
            self.likelihood_.eval()
            means, stds = [], []
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                for start in range(0, len(X), 2048):
                    xb = torch.tensor(X[start:start + 2048], device=self.device_)
                    pred = self.likelihood_(self.model_(xb))
                    means.extend(pred.mean.detach().cpu().numpy().tolist())
                    stds.extend(pred.variance.clamp_min(1e-8).sqrt().detach().cpu().numpy().tolist())
            return np.asarray(means), np.asarray(stds)

        def predict(self, X):
            return self.predict_distribution(X)[0]

    class _DeepKernelGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, input_dim, latent_dim=8):
            super().__init__(train_x, train_y, likelihood)
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, latent_dim),
            )
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=latent_dim)
            )

        def forward(self, x):
            z = self.feature_extractor(x)
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(z), self.covar_module(z)
            )

    class DeepKernelGPRRegressor:
        def __init__(self, latent_dim=8, epochs=100, lr=0.01,
                     max_train_samples=1200, seed=RANDOM_SEED):
            self.latent_dim, self.epochs, self.lr = latent_dim, epochs, lr
            self.max_train_samples, self.seed = max_train_samples, seed

        def predict_distribution(self, X):
            X = np.asarray(X, dtype=np.float32)
            self.model_.eval()
            self.likelihood_.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                xb = torch.tensor(X, device=self.device_)
                pred = self.likelihood_(self.model_(xb))
                return (
                    pred.mean.detach().cpu().numpy(),
                    pred.variance.clamp_min(1e-8).sqrt().detach().cpu().numpy(),
                )

        def predict(self, X):
            return self.predict_distribution(X)[0]
else:
    class _SparseGPModel: pass
    class SparseGPRRegressor: pass
    class _DeepKernelGPModel: pass
    class DeepKernelGPRRegressor: pass


def _jsonable(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def _array(tensor):
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def extract_preprocessing(bundle):
    """Export fitted imputation/standardization/one-hot parameters to JSON."""
    pre = bundle["preprocessor"]
    meta = {
        "feature_columns": list(bundle["feature_columns"]),
        "numeric_features": list(bundle["numeric_features"]),
        "categorical_features": list(bundle["categorical_features"]),
        "transformed_feature_names": list(bundle["feature_names_transformed"]),
        "target_mean": float(bundle["target_scaler"].mean_[0]),
        "target_scale": float(bundle["target_scaler"].scale_[0]),
        "sequence_length": int(bundle.get("sequence_length") or 8),
    }

    numeric = None
    if meta["numeric_features"]:
        pipe = pre.named_transformers_["num"]
        imputer = pipe.named_steps["imputer"]
        scaler = pipe.named_steps["scaler"]
        indicator = getattr(imputer, "indicator_", None)
        indicator_features = (
            np.asarray(getattr(indicator, "features_", []), dtype=int).tolist()
            if indicator is not None else []
        )
        numeric = {
            "imputer_statistics": np.asarray(imputer.statistics_, dtype=float).tolist(),
            "indicator_features": indicator_features,
            "scaler_mean": np.asarray(scaler.mean_, dtype=float).tolist(),
            "scaler_scale": np.asarray(scaler.scale_, dtype=float).tolist(),
        }

    categorical = None
    if meta["categorical_features"]:
        pipe = pre.named_transformers_["cat"]
        imputer = pipe.named_steps["imputer"]
        encoder = pipe.named_steps["onehot"]
        categorical = {
            "imputer_statistics": [str(x) for x in imputer.statistics_],
            "categories": [[str(v) for v in values] for values in encoder.categories_],
        }

    meta["numeric_preprocessing"] = numeric
    meta["categorical_preprocessing"] = categorical
    return meta


def rbf_kernel(X, Z, lengthscale, outputscale=1.0):
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    ls = np.asarray(lengthscale, dtype=np.float64).reshape(-1)
    if ls.size == 1:
        ls = np.repeat(ls, X.shape[1])
    delta = (X[:, None, :] - Z[None, :, :]) / ls[None, None, :]
    d2 = np.einsum("nmd,nmd->nm", delta, delta, optimize=True)
    return float(outputscale) * np.exp(-0.5 * np.maximum(d2, 0.0))


def validate_mean_export(reference, exported, model_name, atol=2e-3, rtol=2e-3):
    reference = np.asarray(reference, dtype=float).reshape(-1)
    exported = np.asarray(exported, dtype=float).reshape(-1)
    max_abs = float(np.max(np.abs(reference - exported))) if len(reference) else 0.0
    if not np.allclose(reference, exported, atol=atol, rtol=rtol):
        raise RuntimeError(
            f"{model_name}: portable export failed parity check; max_abs_error={max_abs:.6g}. "
            "No optimized artifact was accepted."
        )
    return max_abs


def export_gpr(model, out_dir, model_name):
    kernel = model.kernel_
    signal = kernel.k1
    constant = float(signal.k1.constant_value)
    base = signal.k2
    X_train = np.asarray(model.X_train_, dtype=np.float64)
    alpha = np.asarray(model.alpha_, dtype=np.float64).reshape(-1)
    ls = np.asarray(base.length_scale, dtype=np.float64).reshape(-1)
    if hasattr(base, "nu"):
        kind = "matern"
        nu = float(base.nu)
        if nu not in (0.5, 1.5, 2.5):
            raise ValueError(f"Unsupported Matérn nu={nu}; only 0.5, 1.5, 2.5 are exported.")
    else:
        kind, nu = "rbf", None
    np.savez_compressed(
        out_dir / "model.npz", X_train=X_train, alpha=alpha,
        lengthscale=ls, outputscale=np.asarray(constant),
        kernel_kind=np.asarray(kind), nu=np.asarray(nu if nu is not None else -1.0),
    )
    return {"model_format": "exact_gpr_numpy", "kernel": kind, "matern_nu": nu,
            "training_points": int(len(X_train))}


def export_svr(model, out_dir):
    if getattr(model, "kernel", None) != "rbf":
        raise ValueError("Only the trained RBF SVR is supported by the compact exporter.")
    np.savez_compressed(
        out_dir / "model.npz",
        support_vectors=np.asarray(model.support_vectors_, dtype=np.float64),
        dual_coef=np.asarray(model.dual_coef_[0], dtype=np.float64),
        intercept=np.asarray(float(model.intercept_[0])),
        gamma=np.asarray(float(model._gamma)),
    )
    return {"model_format": "rbf_svr_numpy", "support_vectors": int(len(model.support_vectors_))}


def export_random_forest(model, out_dir):
    estimators = list(model.estimators_)
    n_trees = len(estimators)
    max_nodes = max(est.tree_.node_count for est in estimators)
    left = np.full((n_trees, max_nodes), -1, dtype=np.int32)
    right = np.full((n_trees, max_nodes), -1, dtype=np.int32)
    feature = np.full((n_trees, max_nodes), -2, dtype=np.int32)
    threshold = np.zeros((n_trees, max_nodes), dtype=np.float64)
    value = np.zeros((n_trees, max_nodes), dtype=np.float64)
    node_count = np.zeros(n_trees, dtype=np.int32)
    for i, estimator in enumerate(estimators):
        tree = estimator.tree_
        n = tree.node_count
        node_count[i] = n
        left[i, :n] = tree.children_left
        right[i, :n] = tree.children_right
        feature[i, :n] = tree.feature
        threshold[i, :n] = tree.threshold
        value[i, :n] = tree.value[:, 0, 0]
    np.savez_compressed(
        out_dir / "model.npz", left=left, right=right, feature=feature,
        threshold=threshold, value=value, node_count=node_count,
    )
    return {"model_format": "random_forest_arrays", "trees": n_trees, "max_nodes_per_tree": int(max_nodes)}


def export_xgboost(model, out_dir):
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    path = out_dir / "model.ubj"
    booster.save_model(str(path))
    return {"model_format": "xgboost_ubjson", "file": path.name}


def export_onnx(module, out_path, input_shape):
    if torch is None:
        raise ImportError("PyTorch is required to export neural models to ONNX.")
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise ImportError("Install onnx on the training laptop: pip install onnx") from exc
    module = module.to("cpu").eval()
    dummy = torch.zeros(*input_shape, dtype=torch.float32)
    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}
    torch.onnx.export(
        module, dummy, str(out_path),
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=17,
        do_constant_folding=True, dynamo=False,
    )
    return {"onnx_file": out_path.name, "onnx_input_name": "input",
            "onnx_output_name": "output", "onnx_input_shape": [None] + list(input_shape[1:])}


def export_neural(model_name, model, metadata, out_dir):
    if torch is None:
        raise ImportError("PyTorch is required for this model.")
    net = model.model_.to("cpu").eval()
    d = len(metadata["transformed_feature_names"])
    if model_name == "bilstm":
        seq = int(metadata["sequence_length"])
        shape = (1, seq, d)
    else:
        shape = (1, d)
    export_info = export_onnx(net, out_dir / "model.onnx", shape)
    return {"model_format": "onnx_neural", **export_info}


def export_sparse_gpr(model, out_dir):
    if torch is None or gpytorch is None:
        raise ImportError("Sparse GPR export requires torch and gpytorch on the translator machine.")
    gp = model.model_.to("cpu").eval()
    if getattr(model, "likelihood_", None) is not None:
        model.likelihood_.to("cpu").eval()
    strategy = gp.variational_strategy
    inducing = strategy.inducing_points.detach().cpu().double()
    variational = strategy._variational_distribution
    q_mu = variational.variational_mean.detach().cpu().double().reshape(-1)
    with torch.no_grad():
        kuu = gp.covar_module(inducing).to_dense().double()
        jitter = float(getattr(strategy, "jitter_val", 1e-6) or 1e-6)
        eye = torch.eye(kuu.shape[-1], dtype=torch.float64)
        chol = torch.linalg.cholesky(kuu + jitter * eye)
        weights = torch.linalg.solve_triangular(
            chol.transpose(-1, -2), q_mu[:, None], upper=True
        ).squeeze(-1)
        lengthscale = gp.covar_module.base_kernel.lengthscale.detach().cpu().numpy().reshape(-1)
        outputscale = float(gp.covar_module.outputscale.detach().cpu().reshape(-1)[0])
        mean = float(gp.mean_module.constant.detach().cpu().reshape(-1)[0])
    inducing_np = inducing.numpy()
    weights_np = weights.numpy()
    np.savez_compressed(
        out_dir / "model.npz", inducing=inducing_np, weights=weights_np,
        lengthscale=lengthscale, outputscale=np.asarray(outputscale),
        mean=np.asarray(mean),
    )
    sample = np.asarray(model.inducing_points_[:min(12, len(model.inducing_points_))], dtype=np.float32)
    reference = model.predict(sample)
    portable = mean + rbf_kernel(sample, inducing_np, lengthscale, outputscale) @ weights_np
    err = validate_mean_export(reference, portable, "sparse_gpr", atol=5e-3, rtol=5e-3)
    return {"model_format": "sparse_gpr_numpy", "inducing_points": len(inducing_np),
            "parity_max_abs_error_scaled_target": err}


def export_deep_kernel_gpr(model, out_dir):
    if torch is None or gpytorch is None:
        raise ImportError("Deep-kernel GPR export requires torch and gpytorch on the translator machine.")
    gp = model.model_.to("cpu").eval()
    likelihood = model.likelihood_.to("cpu").eval()
    extractor = gp.feature_extractor.to("cpu").eval()
    train_x = gp.train_inputs[0].detach().cpu().float()
    train_y = gp.train_targets.detach().cpu().double().numpy().reshape(-1)
    with torch.no_grad():
        train_z = extractor(train_x).detach().cpu().double().numpy()
        lengthscale = gp.covar_module.base_kernel.lengthscale.detach().cpu().numpy().reshape(-1)
        outputscale = float(gp.covar_module.outputscale.detach().cpu().reshape(-1)[0])
        mean = float(gp.mean_module.constant.detach().cpu().reshape(-1)[0])
        noise = float(likelihood.noise.detach().cpu().reshape(-1)[0])
    K = rbf_kernel(train_z, train_z, lengthscale, outputscale)
    K.flat[::len(K) + 1] += noise + 1e-6
    alpha = np.linalg.solve(K, train_y - mean)
    np.savez_compressed(
        out_dir / "gp_params.npz", train_z=train_z, alpha=alpha,
        lengthscale=lengthscale, outputscale=np.asarray(outputscale),
        mean=np.asarray(mean),
    )
    input_dim = int(train_x.shape[-1])
    onnx_info = export_onnx(extractor, out_dir / "feature_extractor.onnx", (1, input_dim))
    # Check exported GP algebra against the original GPyTorch predictor.
    sample_x = train_x[:min(12, len(train_x))].numpy()
    with torch.no_grad():
        reference = model.predict(sample_x)
        z_sample = extractor(torch.from_numpy(sample_x)).cpu().double().numpy()
    portable = mean + rbf_kernel(z_sample, train_z, lengthscale, outputscale) @ alpha
    err = validate_mean_export(reference, portable, "deep_kernel_gpr", atol=1e-2, rtol=1e-2)
    return {"model_format": "deep_kernel_gpr_numpy_plus_onnx", **onnx_info,
            "gp_training_points": int(len(train_z)),
            "parity_max_abs_error_scaled_target": err}


# gplearn program -> explicit prefix-tree JSON and a portable C expression.
def symbolic_tree(program):
    tokens = list(program)
    cursor = 0

    def parse():
        nonlocal cursor
        if cursor >= len(tokens):
            raise ValueError("Malformed gplearn prefix program.")
        token = tokens[cursor]
        cursor += 1
        if hasattr(token, "arity") and hasattr(token, "name"):
            return {"op": str(token.name), "args": [parse() for _ in range(int(token.arity))]}
        if isinstance(token, (numbers.Integral, np.integer)):
            return {"feature": int(token)}
        if isinstance(token, numbers.Real):
            return {"constant": float(token)}
        raise TypeError(f"Unsupported symbolic token: {token!r}")

    root = parse()
    if cursor != len(tokens):
        raise ValueError("Unconsumed tokens remain in gplearn program.")
    return root


def count_tree_nodes(node):
    if "args" not in node:
        return 1
    return 1 + sum(count_tree_nodes(x) for x in node["args"])


def c_expr(node):
    if "feature" in node: return f"x[{int(node['feature'])}]"
    if "constant" in node: return f"({float(node['constant']):.17g})"
    op = node["op"]
    args = [c_expr(x) for x in node["args"]]
    if op == "add": return f"({args[0]} + {args[1]})"
    if op == "sub": return f"({args[0]} - {args[1]})"
    if op == "mul": return f"({args[0]} * {args[1]})"
    if op == "div": return f"pdiv({args[0]}, {args[1]})"
    if op == "sqrt": return f"psqrt({args[0]})"
    if op == "log": return f"plog({args[0]})"
    if op == "abs": return f"fabs({args[0]})"
    if op == "neg": return f"(-({args[0]}))"
    raise ValueError(f"Unsupported gplearn function: {op}")


def export_symbolic(model, out_dir):
    program_obj = model._program
    tree = symbolic_tree(program_obj.program)
    node_count = int(getattr(program_obj, "length_", count_tree_nodes(tree)))
    (out_dir / "equation.json").write_text(json.dumps(tree, indent=2), encoding="utf-8")
    expr = c_expr(tree)
    c_source = f'''#include <math.h>\n\nstatic inline double pdiv(double a, double b) {{ return fabs(b) > 0.001 ? a / b : 1.0; }}\nstatic inline double psqrt(double a) {{ return sqrt(fabs(a)); }}\nstatic inline double plog(double a) {{ return fabs(a) > 0.001 ? log(fabs(a)) : 0.0; }}\n\n// Returns standardized SOH; apply target_scale and target_mean from metadata.json.\ndouble battery_symbolic_predict(const double *x) {{ return {expr}; }}\n'''
    (out_dir / "equation.c").write_text(c_source, encoding="utf-8")
    return {"model_format": "symbolic_c_json", "equation_node_count": node_count,
            "equation_file": "equation.c", "expression_file": "equation.json"}


def export_one(model_name, force=False):
    source = ROOT / f"{model_name}.joblib"
    if not source.exists():
        raise FileNotFoundError(f"Missing trained artifact: {source}")
    out_dir = OPT_ROOT / model_name
    if out_dir.exists():
        if not force:
            raise FileExistsError(
                f"Optimized export already exists: {out_dir}. Use --force to replace it."
            )
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(source)
    if not isinstance(bundle, dict) or bundle.get("model_name") != model_name:
        raise ValueError(f"Unexpected bundle structure in {source}")
    metadata = extract_preprocessing(bundle)
    model = bundle["model"]

    if model_name in {"gpr_rbf", "gpr_matern", "gpr_ard"}:
        info = export_gpr(model, out_dir, model_name)
    elif model_name == "svr":
        info = export_svr(model, out_dir)
    elif model_name == "random_forest":
        info = export_random_forest(model, out_dir)
    elif model_name == "xgboost":
        info = export_xgboost(model, out_dir)
    elif model_name in {"wide_nn", "deep_nn", "wide_deep_nn", "bilstm"}:
        info = export_neural(model_name, model, metadata, out_dir)
    elif model_name == "sparse_gpr":
        info = export_sparse_gpr(model, out_dir)
    elif model_name == "deep_kernel_gpr":
        info = export_deep_kernel_gpr(model, out_dir)
    elif model_name == "symbolic_regression":
        info = export_symbolic(model, out_dir)
    else:
        raise ValueError(f"No exporter is defined for {model_name}.")

    metadata.update(info)
    metadata.update({
        "format_version": 2,
        "model_name": model_name,
        "source_joblib": source.name,
        "source_joblib_size_bytes": source.stat().st_size,
        "target": bundle.get("target", "soh_pct"),
        "training_cycle_count": int(bundle.get("training_cycle_count", -1)),
        "training_cells": list(bundle.get("training_cells", [])),
        "note": "Original joblib remains unchanged. ONNX engines should be built on target Orin Nano.",
    })
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=_jsonable), encoding="utf-8"
    )
    return {"model": model_name, "status": "ok", "format": info["model_format"],
            "output": str(out_dir)}


def main():
    parser = argparse.ArgumentParser(description="Translate trained baseline .joblib files to Jetson-friendly formats.")
    parser.add_argument("--models", nargs="*", choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument("--force", action="store_true", help="Replace existing optimized export folders.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any requested model fails.")
    args = parser.parse_args()

    OPT_ROOT.mkdir(parents=True, exist_ok=True)
    report = []
    print(f"Source models: {ROOT}")
    print(f"Optimized exports: {OPT_ROOT}")
    print("Original .joblib files will not be modified.\n")

    for name in args.models:
        print(f"[{name}] exporting...", flush=True)
        try:
            result = export_one(name, force=args.force)
            report.append(result)
            print(f"  OK -> {result['output']} ({result['format']})")
        except Exception as exc:
            result = {"model": name, "status": "failed", "error": str(exc)}
            report.append(result)
            print(f"  FAILED: {exc}")
            if os.environ.get("TRANSLATE_TRACEBACK") == "1":
                traceback.print_exc()

    report_path = OPT_ROOT / "translation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    ok = sum(row["status"] == "ok" for row in report)
    failed = len(report) - ok
    print(f"\nTranslation complete: {ok} succeeded, {failed} failed.")
    print(f"Report: {report_path}")
    if failed and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
