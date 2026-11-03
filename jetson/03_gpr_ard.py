#!/usr/bin/env python3
"""Jetson Orin Nano hardware-only benchmark for gpr_ard optimized export."""
from jetson_runtime import main

if __name__ == "__main__":
    main("gpr_ard", __file__)
