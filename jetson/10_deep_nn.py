#!/usr/bin/env python3
"""Jetson Orin Nano hardware-only benchmark for deep_nn optimized export."""
from jetson_runtime import main

if __name__ == "__main__":
    main("deep_nn", __file__)
