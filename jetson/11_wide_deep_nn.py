#!/usr/bin/env python3
"""Jetson Orin Nano hardware-only benchmark for wide_deep_nn optimized export."""
from jetson_runtime import main

if __name__ == "__main__":
    main("wide_deep_nn", __file__)
