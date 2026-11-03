#!/usr/bin/env python3
"""Jetson Orin Nano hardware-only benchmark for bilstm optimized export."""
from jetson_runtime import main

if __name__ == "__main__":
    main("bilstm", __file__)
