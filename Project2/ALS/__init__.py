"""
ALS
=====

Provides an easy way to compute optimal ALS."""
# -*- coding: utf-8 -*-
# File ALS/main.py

try:
    from .main import main
except (ModuleNotFoundError, ImportError):
    from main import main
