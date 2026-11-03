#!/usr/bin/env python3
"""Jetson Orin Nano hardware-only benchmark for sparse_gpr optimized export."""
from jetson_runtime import main

if __name__ == "__main__":
    main("sparse_gpr", __file__)
