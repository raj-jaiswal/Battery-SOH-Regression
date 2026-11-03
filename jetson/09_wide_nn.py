#!/usr/bin/env python3
"""Jetson Orin Nano hardware-only benchmark for wide_nn optimized export."""
from jetson_runtime import main

if __name__ == "__main__":
    main("wide_nn", __file__)
