"""
Baseline
=====

Provides an easy way to compute baseline with mean and user/item bias."""
# -*- coding: utf-8 -*-
# File ALS/main.py

try:
    from .main import main
except (ModuleNotFoundError, ImportError):
    from main import main
