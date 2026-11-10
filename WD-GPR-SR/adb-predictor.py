#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

try:
    import gpytorch
except ImportError as err:
    sys.exit(
        f"Error: gpytorch is required for WD-GPR-SR inference. "
        f"Install dependencies using: pip install -r requirements.txt ({err})"
    )


# ============================================================================
# 1. Wide-and-Deep Sparse Gaussian Process Architecture (matches training bundle)
# ============================================================================

class WideDeepSparseGP(gpytorch.models.ApproximateGP):
    """
    Wide-and-Deep Variational Gaussian Process Regressor.
    Unifies linear wide features and non-linear deep representations with
    variational inducing points for fast, scalable Bayesian inference.
    """

    def __init__(
        self,
        inducing_points: torch.Tensor,
        input_dim: int,
        latent_dim: int = 12,
        kernel_mode: str = "wide_deep",
    ):
        q_dist = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, q_dist, learn_inducing_locations=True
        )
        super().__init__(strategy)
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.kernel_mode = str(kernel_mode)

        if self.kernel_mode in ("wide_deep", "wide_only"):
            self.wide_branch = nn.Linear(self.input_dim, self.latent_dim)
            self.wide_kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.LinearKernel()
            )
        if self.kernel_mode in ("wide_deep", "deep_only"):
            self.deep_branch = nn.Sequential(
                nn.Linear(self.input_dim, 64),
                nn.GELU(),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, self.latent_dim),
            )
            self.deep_kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=self.latent_dim)
            )
        self.mean_module = gpytorch.means.ConstantMean()

    def forward(self, x: torch.Tensor):
        covar = None
        if self.kernel_mode in ("wide_deep", "wide_only"):
            wide_z = self.wide_branch(x)
            covar = self.wide_kernel(wide_z)
        if self.kernel_mode in ("wide_deep", "deep_only"):
            deep_z = self.deep_branch(x)
            deep_covar = self.deep_kernel(deep_z)
            covar = deep_covar if covar is None else covar + deep_covar
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), covar)


def evaluate_symbolic_tree(node: dict, X: np.ndarray) -> np.ndarray:
    """Recursively evaluates the uncertainty-weighted symbolic regression tree."""
    X = np.asarray(X, dtype=float)
    if "feature" in node:
        return X[:, int(node["feature"])]
    if "constant" in node:
        return np.full(len(X), float(node["constant"]), dtype=float)

    args = [evaluate_symbolic_tree(child, X) for child in node.get("args", [])]
    op = node.get("op")

    if op == "add":
        return args[0] + args[1]
    if op == "sub":
        return args[0] - args[1]
    if op == "mul":
        return args[0] * args[1]
    if op == "div":
        # Protected division
        denom = np.where(np.abs(args[1]) > 0.05, args[1], 1.0)
        return np.where(np.abs(args[1]) > 0.05, args[0] / denom, 1.0)
    if op == "sqrt":
        return np.sqrt(np.maximum(args[0], 0.0))
    if op == "log":
        return np.where(args[0] > 0.05, np.log(np.maximum(args[0], 0.05)), 0.0)
    if op == "abs":
        return np.abs(args[0])
    if op == "neg":
        return -args[0]
    raise ValueError(f"Unsupported symbolic operation: {op}")


def make_symbolic_inputs(
    X: np.ndarray,
    gp_std: np.ndarray,
    selected_indices: list[int],
    uncertainty_scale: float,
    include_uncertainty: bool = True,
) -> np.ndarray:
    """Assembles transformed features with normalized GP uncertainty for symbolic residual evaluation."""
    X = np.asarray(X, dtype=np.float32)
    parts = [X[:, np.asarray(selected_indices, dtype=int)]]
    if include_uncertainty:
        scale = max(float(uncertainty_scale), 1e-6)
        u = np.clip(np.asarray(gp_std, dtype=np.float32).reshape(-1) / scale, 0.1, 5.0)
        parts.append(u.reshape(-1, 1))
    return np.column_stack(parts).astype(np.float64)


# ============================================================================
# 2. Model Loader and Inference Engine
# ============================================================================

class WDGPRSRPredictor:
    """Loads wd_gpr_sr.joblib and executes calibrated SOH point & interval predictions."""

    def __init__(self, model_path: Path):
        if not model_path.is_file():
            raise FileNotFoundError(f"WD-GPR-SR model file not found at: {model_path}")

        self.model_path = model_path
        self.bundle = joblib.load(model_path)
        self.preprocessor = self.bundle["preprocessor"]
        self.target_scaler = self.bundle["target_scaler"]
        self.gp_config = self.bundle["gp_config"]
        self.feature_columns = self.bundle["feature_columns"]
        self.symbolic_feature_indices = self.bundle["symbolic_feature_indices"]
        self.symbolic_tree = self.bundle["symbolic_tree"]
        self.uncertainty_scale = float(self.bundle["uncertainty_scale"])
        self.symbolic_residual_std_scaled = float(
            self.bundle["symbolic_residual_std_scaled"]
        )

        # Build PyTorch GP model and Gaussian Likelihood
        inducing = torch.as_tensor(self.bundle["inducing_points"], dtype=torch.float32)
        self.gp_model = WideDeepSparseGP(
            inducing_points=inducing,
            input_dim=self.gp_config["input_dim"],
            latent_dim=self.gp_config["latent_dim"],
            kernel_mode=self.gp_config["kernel_mode"],
        )
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        self.gp_model.load_state_dict(self.bundle["gp_state_dict"])
        self.likelihood.load_state_dict(self.bundle["likelihood_state_dict"])
        self.gp_model.eval()
        self.likelihood.eval()

    def predict(self, feature_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs WD-GPR-SR inference on a dataframe containing feature_columns.
        Returns point estimate, uncertainty intervals, GP mean, and symbolic correction.
        """
        X_trans = self.preprocessor.transform(feature_df[self.feature_columns])
        X_tensor = torch.as_tensor(X_trans, dtype=torch.float32)

        with torch.no_grad():
            pred_dist = self.likelihood(self.gp_model(X_tensor))
            gp_mean = pred_dist.mean.detach().cpu().numpy()
            gp_std = pred_dist.variance.clamp_min(1e-8).sqrt().detach().cpu().numpy()

        sr_X = make_symbolic_inputs(
            X_trans,
            gp_std,
            self.symbolic_feature_indices,
            self.uncertainty_scale,
            include_uncertainty=True,
        )
        sr_correction = evaluate_symbolic_tree(self.symbolic_tree, sr_X)

        # Combined scaled prediction and aleatoric + epistemic uncertainty
        pred_scaled = gp_mean + sr_correction
        total_std_scaled = np.sqrt(gp_std**2 + self.symbolic_residual_std_scaled**2)

        # Invert target scaling
        target_scale = float(self.target_scaler.scale_[0])
        target_mean = float(self.target_scaler.mean_[0])

        pred_soh = pred_scaled * target_scale + target_mean
        total_std = total_std_scaled * target_scale
        gp_mean_soh = gp_mean * target_scale + target_mean
        sr_correction_soh = sr_correction * target_scale

        ci_lower = pred_soh - 1.96 * total_std
        ci_upper = pred_soh + 1.96 * total_std

        return {
            "soh_pct": float(pred_soh[0]),
            "std_pct": float(total_std[0]),
            "ci_95_lower": float(ci_lower[0]),
            "ci_95_upper": float(ci_upper[0]),
            "gp_prior_soh": float(gp_mean_soh[0]),
            "sr_residual_soh": float(sr_correction_soh[0]),
            "gp_std_scaled": float(gp_std[0]),
        }


# ============================================================================
# 3. Phone Telemetry Data Structures and ADB Interface
# ============================================================================

@dataclass
class TelemetrySample:
    timestamp: float
    voltage_V: float
    current_A: float
    temperature_C: float
    soc_pct: float
    charging: bool


@dataclass
class DeviceMetadata:
    serial: str
    model: str
    brand: str
    manufacturer: str
    android_version: str
    battery_technology: str
    battery_health_status: str
    hardware_cycle_count: Optional[int]
    charge_full_design_uah: Optional[float]
    charge_full_uah: Optional[float]


class ADBTelemetryCollector:
    """Manages ADB connection, querying, and sampling from an Android smartphone."""

    def __init__(self, serial: Optional[str] = None):
        self.adb_bin = self._find_adb()
        self.serial = serial
        self._target_args = ["-s", self.serial] if self.serial else []

    @staticmethod
    def _find_adb() -> str:
        bin_path = shutil.which("adb")
        if bin_path:
            return bin_path
        # Check standard Android SDK locations
        home = Path.home()
        for cand in [
            home / "Android" / "Sdk" / "platform-tools" / "adb",
            home / "android-sdk" / "platform-tools" / "adb",
            Path("/usr/bin/adb"),
            Path("/usr/local/bin/adb"),
        ]:
            if cand.is_file() and os.access(cand, os.X_OK):
                return str(cand)
        raise FileNotFoundError(
            "ADB executable ('adb') not found. Please install Android platform-tools "
            "or ensure 'adb' is in your system PATH."
        )

    def run_cmd(self, args: List[str], timeout: float = 10.0) -> Tuple[int, str, str]:
        cmd = [self.adb_bin] + self._target_args + args
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return res.returncode, res.stdout.strip(), res.stderr.strip()
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as exc:
            return -2, "", str(exc)

    def list_devices(self) -> List[Dict[str, str]]:
        code, out, _ = self.run_cmd(["devices"])
        devices = []
        if code != 0:
            return devices
        lines = out.splitlines()
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 2:
                devices.append({"serial": parts[0], "state": parts[1]})
        return devices

    def wait_for_device(self, timeout: float = 30.0) -> str:
        """Waits until a functional Android device is attached and authorized."""
        start = time.time()
        print("Checking for connected Android devices over ADB...")
        while time.time() - start < timeout:
            devs = self.list_devices()
            online = [d for d in devs if d["state"] == "device"]
            if online:
                chosen = online[0]["serial"]
                if not self.serial:
                    self.serial = chosen
                    self._target_args = ["-s", self.serial]
                print(f"[OK] Connected to device: {self.serial}")
                return self.serial

            unauth = [d for d in devs if d["state"] == "unauthorized"]
            if unauth:
                print(
                    "[!] Device detected but unauthorized. Please check your phone screen "
                    "and accept the 'Allow USB debugging?' RSA prompt."
                )

            time.sleep(1.5)

        raise TimeoutError(
            f"No authorized Android device found within {timeout:.0f} seconds.\n"
            "Steps to connect:\n"
            "  1. Connect your phone via USB.\n"
            "  2. Enable Developer Options -> USB Debugging.\n"
            "  3. Authorize the PC fingerprint prompt on your phone.\n"
            "  4. Re-run this script (or use --simulate for simulated telemetry)."
        )

    def query_device_metadata(self) -> DeviceMetadata:
        """Fetches device model, build properties, and battery hardware capabilities."""
        code, model, _ = self.run_cmd(["shell", "getprop", "ro.product.model"])
        _, brand, _ = self.run_cmd(["shell", "getprop", "ro.product.brand"])
        _, mfg, _ = self.run_cmd(["shell", "getprop", "ro.product.manufacturer"])
        _, ver, _ = self.run_cmd(
            ["shell", "getprop", "ro.build.version.release"]
        )

        # Composite battery query
        bat_script = (
            "dumpsys battery; echo '===SYSFS==='; "
            "cat /sys/class/power_supply/battery/cycle_count 2>/dev/null; echo '---'; "
            "cat /sys/class/power_supply/battery/charge_full 2>/dev/null; echo '---'; "
            "cat /sys/class/power_supply/battery/charge_full_design 2>/dev/null; echo '---'; "
            "cat /sys/class/power_supply/bms/cycle_count 2>/dev/null; echo '---'; "
            "cat /sys/class/power_supply/bms/charge_full 2>/dev/null"
        )
        _, raw_bat, _ = self.run_cmd(["shell", bat_script], timeout=6.0)

        tech = "Li-ion"
        health = "Good"
        cycle_count = None
        charge_full = None
        charge_full_design = None

        if "===SYSFS===" in raw_bat:
            dumpsys_part, sysfs_part = raw_bat.split("===SYSFS===", 1)
        else:
            dumpsys_part, sysfs_part = raw_bat, ""

        # Parse dumpsys battery
        for line in dumpsys_part.splitlines():
            line_str = line.strip()
            if line_str.lower().startswith("technology:"):
                tech = line_str.split(":", 1)[1].strip() or tech
            elif line_str.lower().startswith("health:"):
                # Usually health: 2 (GOOD)
                health_val = line_str.split(":", 1)[1].strip()
                health_map = {
                    "1": "Unknown",
                    "2": "Good",
                    "3": "Overheat",
                    "4": "Dead",
                    "5": "Over Voltage",
                    "6": "Unspecified Failure",
                    "7": "Cold",
                }
                health = health_map.get(health_val, health_val)
            elif "cycle count" in line_str.lower() or "cycle_count" in line_str.lower():
                nums = re.findall(r"\d+", line_str)
                if nums and cycle_count is None:
                    cycle_count = int(nums[-1])

        # Parse sysfs parts
        sysfs_tokens = [t.strip() for t in sysfs_part.split("---")]
        if len(sysfs_tokens) >= 1 and sysfs_tokens[0].isdigit():
            cycle_count = int(sysfs_tokens[0])
        if len(sysfs_tokens) >= 2 and sysfs_tokens[1].isdigit():
            charge_full = float(sysfs_tokens[1])
        if len(sysfs_tokens) >= 3 and sysfs_tokens[2].isdigit():
            charge_full_design = float(sysfs_tokens[2])
        if len(sysfs_tokens) >= 4 and sysfs_tokens[3].isdigit() and cycle_count is None:
            cycle_count = int(sysfs_tokens[3])
        if len(sysfs_tokens) >= 5 and sysfs_tokens[4].isdigit() and charge_full is None:
            charge_full = float(sysfs_tokens[4])

        return DeviceMetadata(
            serial=self.serial or "unknown",
            model=model or "Android Device",
            brand=brand or "Unknown",
            manufacturer=mfg or "Unknown",
            android_version=ver or "Unknown",
            battery_technology=tech,
            battery_health_status=health,
            hardware_cycle_count=cycle_count,
            charge_full_design_uah=charge_full_design,
            charge_full_uah=charge_full,
        )

    def sample_instantaneous_telemetry(self) -> TelemetrySample:
        """
        Executes a composite shell command to extract instantaneous battery voltage,
        current, temperature, and level in a single round-trip.
        """
        cmd_str = (
            "dumpsys battery; echo '===CURR==='; "
            "cat /sys/class/power_supply/battery/current_now 2>/dev/null || "
            "cat /sys/class/power_supply/bms/current_now 2>/dev/null; echo '---'; "
            "cat /sys/class/power_supply/battery/voltage_now 2>/dev/null; echo '---'; "
            "cat /sys/class/power_supply/battery/temp 2>/dev/null"
        )
        _, out, _ = self.run_cmd(["shell", cmd_str], timeout=4.0)

        dumpsys_text = out
        sysfs_text = ""
        if "===CURR===" in out:
            dumpsys_text, sysfs_text = out.split("===CURR===", 1)

        # Parse dumpsys fields
        raw_voltage = None
        raw_temp = None
        raw_level = None
        status = None
        max_charging_current = None

        for line in dumpsys_text.splitlines():
            line_str = line.strip().lower()
            if line_str.startswith("voltage:"):
                val = re.findall(r"\d+", line_str)
                if val:
                    raw_voltage = float(val[0])
            elif line_str.startswith("temperature:"):
                val = re.findall(r"\d+", line_str)
                if val:
                    raw_temp = float(val[0])
            elif line_str.startswith("level:"):
                val = re.findall(r"\d+", line_str)
                if val:
                    raw_level = float(val[0])
            elif line_str.startswith("status:"):
                # 2 = charging, 3 = discharging, 4 = not charging, 5 = full
                val = re.findall(r"\d+", line_str)
                if val:
                    status = int(val[0])
            elif "max charging current" in line_str:
                val = re.findall(r"\d+", line_str)
                if val:
                    max_charging_current = float(val[0])

        # Parse sysfs lines
        sysfs_parts = [p.strip() for p in sysfs_text.split("---")]
        sysfs_curr_raw = sysfs_parts[0] if len(sysfs_parts) > 0 else ""
        sysfs_volt_raw = sysfs_parts[1] if len(sysfs_parts) > 1 else ""
        sysfs_temp_raw = sysfs_parts[2] if len(sysfs_parts) > 2 else ""

        # 1. Translate Voltage -> Volts
        # Priority: sysfs voltage_now if valid, else dumpsys voltage (mV)
        volt_v = 3.85
        if sysfs_volt_raw.lstrip("-").isdigit():
            val = float(sysfs_volt_raw)
            if val > 100_000:
                volt_v = val / 1_000_000.0  # microvolts -> V
            elif val > 1_000:
                volt_v = val / 1_000.0  # millivolts -> V
            elif 2.0 <= val <= 5.0:
                volt_v = val
        elif raw_voltage is not None:
            if raw_voltage > 100_000:
                volt_v = raw_voltage / 1_000_000.0
            elif raw_voltage > 1_000:
                volt_v = raw_voltage / 1_000.0
            else:
                volt_v = raw_voltage

        # 2. Translate Current -> Amperes
        # Note: current_now can be positive or negative depending on kernel sign convention.
        curr_a = 0.45
        has_curr = False
        if sysfs_curr_raw.lstrip("-").isdigit():
            val = float(sysfs_curr_raw)
            if abs(val) > 10_000:
                curr_a = val / 1_000_000.0  # uA -> A
                has_curr = True
            elif abs(val) > 10:
                curr_a = val / 1_000.0  # mA -> A
                has_curr = True
            elif 0.01 <= abs(val) <= 10.0:
                curr_a = val
                has_curr = True

        if not has_curr:
            # Fallback based on charging state
            if status == 2 and max_charging_current is not None and max_charging_current > 0:
                curr_a = (
                    max_charging_current / 1_000_000.0
                    if max_charging_current > 10_000
                    else max_charging_current / 1000.0
                )
            elif status == 2:
                curr_a = 1.0  # Typical standard USB 1A charge
            else:
                curr_a = -0.45  # Typical active screen phone discharge: ~450 mA

        # 3. Translate Temperature -> Celsius
        temp_c = 28.5
        if sysfs_temp_raw.lstrip("-").isdigit():
            val = float(sysfs_temp_raw)
            if val > 1_000:
                temp_c = val / 1_000.0  # milli-C -> C
            elif val > 100:
                temp_c = val / 10.0  # tenths of C -> C
            elif 0.0 <= val <= 80.0:
                temp_c = val
        elif raw_temp is not None:
            if raw_temp > 1_000:
                temp_c = raw_temp / 1_000.0
            elif raw_temp > 100:
                temp_c = raw_temp / 10.0
            else:
                temp_c = raw_temp

        # 4. State of Charge
        soc_pct = float(raw_level) if raw_level is not None else 80.0
        charging = (status == 2) or (curr_a > 0.05)

        return TelemetrySample(
            timestamp=time.time(),
            voltage_V=volt_v,
            current_A=curr_a,
            temperature_C=temp_c,
            soc_pct=soc_pct,
            charging=charging,
        )


# ============================================================================
# 4. Simulated Telemetry Generator (for testing without a connected phone)
# ============================================================================

class SimulatedTelemetryCollector:
    """Generates realistic phone battery discharge telemetry for rapid testing and dry runs."""

    def __init__(self, target_soh: float = 88.5, cycle_count: int = 145):
        self.target_soh = target_soh
        self.cycle_count = cycle_count
        self.soc = 76.0
        self.base_v = 3.98
        self.base_temp = 29.2

    def query_device_metadata(self) -> DeviceMetadata:
        return DeviceMetadata(
            serial="SIMULATED-PIXEL-7A",
            model="Pixel 7a (Simulated)",
            brand="Google",
            manufacturer="Google",
            android_version="14",
            battery_technology="Li-ion Polymer",
            battery_health_status="Good",
            hardware_cycle_count=self.cycle_count,
            charge_full_design_uah=4385000.0,
            charge_full_uah=4385000.0 * (self.target_soh / 100.0),
        )

    def sample_instantaneous_telemetry(self) -> TelemetrySample:
        # Realistic slight jitter on voltage, current and temperature
        v_noise = float(np.random.normal(0, 0.003))
        i_noise = float(np.random.normal(0, 0.025))
        t_noise = float(np.random.normal(0, 0.05))

        return TelemetrySample(
            timestamp=time.time(),
            voltage_V=round(self.base_v + v_noise, 4),
            current_A=round(-0.48 + i_noise, 4),
            temperature_C=round(self.base_temp + t_noise, 2),
            soc_pct=self.soc,
            charging=False,
        )


# ============================================================================
# 5. Telemetry Translator: Phone Live Window -> Full Cycle Degradation Features
# ============================================================================

class TelemetryTranslator:
    """
    Translates short-window (20-30s) instantaneous smartphone telemetry
    into the 15 numeric + 2 categorical features expected by the WD-GPR-SR model.
    """

    # Reference discharge curve parameters calibrated to Li-ion pouch cells (CALCE CS2)
    # Calibrated to real lab discharge plateau: nominal ~3.73V mean, ~0.21V std, 2.70V-4.10V span
    REF_V_MIN = 2.6998
    REF_V_MAX = 4.1020
    REF_V_MEAN = 3.7297
    REF_V_STD = 0.2113
    REF_V_MEDIAN = 3.7217
    REF_V_Q10 = 3.5602
    REF_V_Q90 = 3.9840

    @staticmethod
    def _nominal_ocv(soc_norm: float) -> float:
        """Standard Li-ion Open Circuit Voltage (OCV) empirical function for 0 <= soc <= 1."""
        s = float(np.clip(soc_norm, 0.005, 0.995))
        return (
            3.42
            + 0.78 * s
            - 0.04 / (s**0.5 + 0.02)
            + 0.05 * math.log(s / (1.0 - s + 1e-4))
        )

    @classmethod
    def translate(
        cls,
        samples: List[TelemetrySample],
        metadata: DeviceMetadata,
        mode: str = "cycle",
        manual_cycles: Optional[int] = None,
        battery_family: str = "CS2",
    ) -> pd.DataFrame:
        """
        Translates a list of TelemetrySample records into a single-row DataFrame
        ready for WD-GPR-SR model consumption.

        Parameters:
          mode: 'cycle' (Virtual Cycle Synthesis) or 'direct' (raw window statistics)
          manual_cycles: Optional manual override for cycle count
          battery_family: Default 'CS2' (matches Li-ion pouch cells used in phones)
        """
        if not samples:
            raise ValueError("No telemetry samples collected to translate.")

        voltages = np.array([s.voltage_V for s in samples], dtype=float)
        currents_abs = np.array([abs(s.current_A) for s in samples], dtype=float)
        temps = np.array([s.temperature_C for s in samples], dtype=float)
        socs = np.array([s.soc_pct for s in samples], dtype=float)

        mean_v = float(np.mean(voltages))
        mean_soc = float(np.mean(socs)) / 100.0
        mean_i = float(np.mean(currents_abs))
        std_i = float(np.std(currents_abs))
        median_i = float(np.median(currents_abs))

        mean_t = float(np.mean(temps))
        std_t = float(np.std(temps))
        min_t = float(np.min(temps))
        max_t = float(np.max(temps))

        # Determine cycle count: manual override > hardware report > estimation > default
        if manual_cycles is not None and manual_cycles > 0:
            cycle_order = float(manual_cycles)
        elif metadata.hardware_cycle_count is not None and metadata.hardware_cycle_count > 0:
            cycle_order = float(metadata.hardware_cycle_count)
        elif (
            metadata.charge_full_design_uah is not None
            and metadata.charge_full_uah is not None
            and metadata.charge_full_design_uah > 0
        ):
            # Estimate from Coulomb-counter degradation (~0.05% loss per full equivalent cycle)
            deg_fraction = max(
                0.0,
                1.0 - (metadata.charge_full_uah / metadata.charge_full_design_uah),
            )
            cycle_order = float(max(10.0, min(deg_fraction / 0.0005, 450.0)))
        else:
            # Default to representative cycle order matching training mean (105)
            cycle_order = 105.0

        # Dataset attribution matching trained model structure
        if battery_family in ("CS2", "CX2"):
            source_dataset = "CALCE"
            # CALCE training data had unmeasured temperature (handled by model's median imputer)
            temp_mean_val = np.nan
            temp_std_val = np.nan
            temp_min_val = np.nan
            temp_max_val = np.nan
        else:
            source_dataset = "NASA"
            temp_mean_val = mean_t
            temp_std_val = max(std_t, 0.05)
            temp_min_val = min_t
            temp_max_val = max_t

        if mode == "direct":
            # Direct raw window statistics (useful for baseline inspection)
            v_mean = mean_v
            v_std = max(float(np.std(voltages)), 0.001)
            v_min = float(np.min(voltages))
            v_max = float(np.max(voltages))
            v_median = float(np.median(voltages))
            v_q10 = float(np.quantile(voltages, 0.10))
            v_q90 = float(np.quantile(voltages, 0.90))
        else:
            # Physics-Informed Virtual Cycle Synthesis:
            # Reconstructs the full-cycle discharge curve taking into account
            # the measured operating point's overpotential and internal resistance drop.
            expected_ocv = cls._nominal_ocv(mean_soc)
            delta_v = mean_v - expected_ocv  # Polarization / health offset

            v_mean = float(cls.REF_V_MEAN + delta_v)
            v_std = float(cls.REF_V_STD)
            v_min = float(cls.REF_V_MIN + delta_v)
            v_max = float(cls.REF_V_MAX + delta_v)
            v_median = float(cls.REF_V_MEDIAN + delta_v)
            v_q10 = float(cls.REF_V_Q10 + delta_v)
            v_q90 = float(cls.REF_V_Q90 + delta_v)

        row = {
            "cycle_order": cycle_order,
            "voltage_mean_V": v_mean,
            "voltage_std_V": v_std,
            "voltage_min_V": v_min,
            "voltage_max_V": v_max,
            "voltage_median_V": v_median,
            "voltage_q10_V": v_q10,
            "voltage_q90_V": v_q90,
            "current_abs_mean_A": max(mean_i, 0.05),
            "current_abs_std_A": max(std_i, 0.001),
            "current_abs_median_A": max(median_i, 0.05),
            "temperature_mean_C": temp_mean_val,
            "temperature_std_C": temp_std_val,
            "temperature_min_C": temp_min_val,
            "temperature_max_C": temp_max_val,
            "source_dataset": source_dataset,
            "battery_family": battery_family,
        }

        return pd.DataFrame([row])


# ============================================================================
# 6. Live Sampling Runner and Reporting CLI
# ============================================================================

def format_health_badge(soh_pct: float) -> str:
    if soh_pct >= 90.0:
        return "[EXCELLENT] Healthy battery with minimal degradation"
    elif soh_pct >= 80.0:
        return "[GOOD] Normal battery condition"
    elif soh_pct >= 70.0:
        return "[DEGRADED] Noticeable capacity loss; monitor closely"
    else:
        return "[CRITICAL] Severe degradation; battery replacement recommended"


def print_progress_bar(
    current: int,
    total: int,
    v: float,
    i: float,
    t: float,
    soc: float,
    charging: bool,
    bar_len: int = 25,
):
    frac = current / total
    filled = int(round(bar_len * frac))
    bar = "=" * filled + (">" if filled < bar_len else "") + "." * (bar_len - filled - 1)
    status_str = "CHG" if charging else "DIS"
    sys.stdout.write(
        f"\r[{current:02d}/{total:02d}s] [{bar}] "
        f"V: {v:.3f}V | I: {i:+.3f}A ({status_str}) | T: {t:4.1f}\u00b0C | SOC: {soc:3.0f}%"
    )
    sys.stdout.flush()


def find_model_path() -> Path:
    candidates = [
        Path(__file__).resolve().parent.parent / "models" / "wd_gpr_sr.joblib",
        Path(__file__).resolve().parent / "models" / "wd_gpr_sr.joblib",
        Path.cwd() / "models" / "wd_gpr_sr.joblib",
        Path.cwd().parent / "models" / "wd_gpr_sr.joblib",
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    return candidates[0]


def run_live_prediction(args: argparse.Namespace):
    model_path = find_model_path()

    print("\n" + "=" * 70)
    print(" LIVE SOH PREDICTOR FOR PHONE LI-ION BATTERY (WD-GPR-SR)")
    print("=" * 70)

    # 1. Initialize Predictor
    print(f"Loading pretrained WD-GPR-SR model from: {model_path.name} ...")
    predictor = WDGPRSRPredictor(model_path)
    print("[OK] Model loaded successfully.")

    # 2. Setup Telemetry Collector
    if args.simulate:
        print("[!] Running in SIMULATION mode (--simulate flag enabled).")
        collector = SimulatedTelemetryCollector(
            target_soh=args.sim_soh,
            cycle_count=args.cycles or 145,
        )
    else:
        adb_collector = ADBTelemetryCollector(serial=args.device)
        try:
            adb_collector.wait_for_device(timeout=args.wait_timeout)
        except Exception as e:
            print(f"\n[ERROR] {e}")
            print("\nTip: Use --simulate to test the model right now without a phone.")
            sys.exit(1)
        collector = adb_collector

    # 3. Query Device Hardware Metadata
    meta = collector.query_device_metadata()
    print("\n--- CONNECTED DEVICE TELEMETRY ---")
    print(f"  Model:          {meta.brand} {meta.model} (Android {meta.android_version})")
    print(f"  Serial:         {meta.serial}")
    print(f"  Technology:     {meta.battery_technology}")
    print(f"  OS Health:      {meta.battery_health_status}")
    if meta.hardware_cycle_count is not None:
        print(f"  Hardware Cycles:{meta.hardware_cycle_count}")
    else:
        print(f"  Hardware Cycles:Not exposed by kernel (using default/override)")
    if meta.charge_full_design_uah and meta.charge_full_uah:
        hw_soh = (meta.charge_full_uah / meta.charge_full_design_uah) * 100.0
        print(
            f"  Coulomb Metric: {meta.charge_full_uah / 1000.0:.0f} / "
            f"{meta.charge_full_design_uah / 1000.0:.0f} mAh (HW SOH ~ {hw_soh:.1f}%)"
        )

    # 4. Telemetry Collection Loop
    duration = int(args.duration)
    interval = float(args.interval)
    total_steps = max(1, int(round(duration / interval)))

    print(f"\nSampling live telemetry for {duration} seconds (interval: {interval}s)...")
    samples: List[TelemetrySample] = []
    start_time = time.time()

    for step in range(1, total_steps + 1):
        sample = collector.sample_instantaneous_telemetry()
        samples.append(sample)
        print_progress_bar(
            current=step,
            total=total_steps,
            v=sample.voltage_V,
            i=sample.current_A,
            t=sample.temperature_C,
            soc=sample.soc_pct,
            charging=sample.charging,
        )
        time.sleep(interval)

    print("\n[OK] Telemetry sampling completed.")

    # 5. Translate Telemetry
    translated_df = TelemetryTranslator.translate(
        samples=samples,
        metadata=meta,
        mode=args.mode,
        manual_cycles=args.cycles,
        battery_family=args.battery_family,
    )

    # 6. Execute WD-GPR-SR Inference
    result = predictor.predict(translated_df)

    # 7. Print Comprehensive Report
    avg_v = float(np.mean([s.voltage_V for s in samples]))
    avg_i = float(np.mean([abs(s.current_A) for s in samples]))
    avg_t = float(np.mean([s.temperature_C for s in samples]))
    last_soc = samples[-1].soc_pct
    cycles_used = translated_df["cycle_order"].iloc[0]

    print("\n" + "=" * 70)
    print("                    BATTERY HEALTH REPORT")
    print("=" * 70)
    print(f" Device:               {meta.brand} {meta.model}")
    print(f" Sampling Window:      {duration}s ({len(samples)} samples)")
    print(f" Live Operating Point: V_avg={avg_v:.3f}V | I_avg={avg_i:.3f}A | T_avg={avg_t:.1f}\u00b0C | SOC={last_soc:.0f}%")
    print(f" Aging Cycle Basis:    {cycles_used:.0f} cycles")
    print(f" Translation Mode:     {args.mode} (Li-ion Pouch Cell Equivalent)")
    print("-" * 70)
    print(f" PREDICTED SOH:        {result['soh_pct']:.2f}%")
    print(f" 95% CREDIBLE INTERVAL:[{result['ci_95_lower']:.2f}%, {result['ci_95_upper']:.2f}%] (\u00b1{1.96 * result['std_pct']:.2f}%)")
    print(f" GP Latent Prior Mean: {result['gp_prior_soh']:.2f}%")
    print(f" Symbolic Correction:  {result['sr_residual_soh']:+.2f}%")
    print(f" Status Assessment:    {format_health_badge(result['soh_pct'])}")
    print("=" * 70)

    # 8. Optional Save to JSON
    if args.save:
        save_path = Path(args.save)
        export_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": asdict(meta),
            "telemetry_summary": {
                "duration_seconds": duration,
                "samples_count": len(samples),
                "voltage_mean_V": avg_v,
                "current_abs_mean_A": avg_i,
                "temperature_mean_C": avg_t,
                "soc_pct": last_soc,
                "cycle_order": cycles_used,
            },
            "translation_features": translated_df.to_dict(orient="records")[0],
            "prediction": result,
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)
        print(f"[OK] Full diagnostic report saved to: {save_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live Battery SOH Predictor using ADB Telemetry and WD-GPR-SR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=int,
        default=25,
        help="Telemetry collection window duration in seconds (20-30s recommended).",
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=1.0,
        help="Sampling interval in seconds.",
    )
    parser.add_argument(
        "--device",
        "-s",
        type=str,
        default=None,
        help="ADB device serial if multiple phones are connected.",
    )
    parser.add_argument(
        "--cycles",
        "-c",
        type=int,
        default=None,
        help="Manual battery cycle count override if phone kernel does not report cycles.",
    )
    parser.add_argument(
        "--mode",
        choices=["cycle", "direct"],
        default="cycle",
        help="Translation mode: 'cycle' (Virtual Cycle Synthesis) or 'direct' (raw window).",
    )
    parser.add_argument(
        "--battery-family",
        type=str,
        default="CS2",
        choices=["CS2", "CX2", "NASA_18650"],
        help="Target reference Li-ion battery chemistry/family ('CS2' is pouch cell like phones).",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run in simulation mode without requiring a physical phone.",
    )
    parser.add_argument(
        "--sim-soh",
        type=float,
        default=88.5,
        help="Target SOH for simulation mode.",
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=30.0,
        help="Max seconds to wait for phone connection over ADB before failing.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save telemetry and prediction report as JSON.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run_live_prediction(cli_args)
