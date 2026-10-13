"""
SHADOW — Streamlit Community Cloud Entrypoint
Machine Learning & AI Battle Arena Dashboard
"""
from pathlib import Path
import sys

# Ensure repository root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from shadow_arena.app import main

if __name__ == "__main__":
    main()
