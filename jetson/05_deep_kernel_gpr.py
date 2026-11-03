#!/usr/bin/env python3
"""Jetson Orin Nano hardware-only benchmark for deep_kernel_gpr optimized export."""
from jetson_runtime import main

if __name__ == "__main__":
    main("deep_kernel_gpr", __file__)
