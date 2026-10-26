"""Single-command runner for reproducible holdout evaluation."""

import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from reproducible_eval import run_evaluation_cli

if __name__ == "__main__":
    run_evaluation_cli()
