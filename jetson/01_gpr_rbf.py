#!/usr/bin/env python3
"""Optimized Jetson Orin Nano inference benchmark for gpr_rbf."""
from jetson_runtime import main

if __name__ == "__main__":
    main("gpr_rbf", __file__)
