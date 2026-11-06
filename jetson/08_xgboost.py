#!/usr/bin/env python3
"""Optimized Jetson Orin Nano inference benchmark for xgboost."""
from jetson_runtime import main

if __name__ == "__main__":
    main("xgboost", __file__)
