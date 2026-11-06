#!/usr/bin/env python3
"""Optimized Jetson Orin Nano inference benchmark for deep_nn."""
from jetson_runtime import main

if __name__ == "__main__":
    main("deep_nn", __file__)
