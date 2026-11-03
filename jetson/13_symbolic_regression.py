#!/usr/bin/env python3
"""Jetson Orin Nano hardware-only benchmark for symbolic_regression optimized export."""
from jetson_runtime import main

if __name__ == "__main__":
    main("symbolic_regression", __file__)
