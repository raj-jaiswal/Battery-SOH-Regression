#!/usr/bin/env python3
"""Optimized Jetson Orin Nano inference benchmark for random_forest."""
from jetson_runtime import main

if __name__ == "__main__":
    main("random_forest", __file__)
