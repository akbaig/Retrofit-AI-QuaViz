"""Locate the challenge package root for training and scoring."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Default: this folder lives at retrofit_starter/solution/final_model
SOURCE = Path(os.environ.get("QUAVIZ_CHALLENGE_DIR", HERE.parents[1])).expanduser().resolve()
