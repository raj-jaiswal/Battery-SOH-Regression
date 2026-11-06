#!/usr/bin/env python3
"""Optimized Jetson Orin Nano inference benchmark for sparse_gpr."""
from jetson_runtime import main

if __name__ == "__main__":
    main("sparse_gpr", __file__)
