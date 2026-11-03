#!/usr/bin/env python3
"""Jetson Orin Nano hardware-only benchmark for xgboost optimized export."""
from jetson_runtime import main

if __name__ == "__main__":
    main("xgboost", __file__)
