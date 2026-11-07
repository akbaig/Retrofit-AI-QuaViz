"""Locate the user's unmodified challenge bundle for training and scoring."""
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
SOURCE=Path(os.environ.get('QUAVIZ_CHALLENGE_DIR',str(HERE.parents[1]/
    'BusinessTurku_SinceAI2026_FINAL_BUILDER_PUBLISH/RETROFIT_AI_PARTICIPANTS_2026'))).expanduser().resolve()
