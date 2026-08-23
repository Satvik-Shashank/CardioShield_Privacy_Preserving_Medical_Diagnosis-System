"""
conftest.py
───────────
Pytest configuration ensuring workspace root is on sys.path.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
