"""
Surprize
=====

Provides an easy way to compute ensemble methods with the Surprize library."""
# -*- coding: utf-8 -*-
# File ALS/main.py

try:
    from .main import main, tester
except (ModuleNotFoundError, ImportError):
    from main import main, tester
