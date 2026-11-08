"""Resolve challenge kit + frozen model locations; put them on sys.path."""
from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]  # app/backend
APP_ROOT = BACKEND_ROOT.parent  # app/
REPO_ROOT = APP_ROOT.parent  # Retrofit-AI-QuaViz/


def challenge_dir() -> Path:
    raw = os.environ.get("QUAVIZ_CHALLENGE_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return (REPO_ROOT / "retrofit_starter").resolve()


def final_model_dir(challenge: Path | None = None) -> Path:
    root = challenge or challenge_dir()
    return (root / "solution" / "final_model").resolve()


def data_dir(challenge: Path | None = None) -> Path:
    root = challenge or challenge_dir()
    override = os.environ.get("QUAVIZ_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (root / "data" / "full").resolve()


def model_path(challenge: Path | None = None) -> Path:
    override = os.environ.get("QUAVIZ_MODEL_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return final_model_dir(challenge) / "model.joblib"


def policy_path(challenge: Path | None = None) -> Path:
    override = os.environ.get("QUAVIZ_POLICY_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return final_model_dir(challenge) / "policy.json"


def bootstrap_imports(challenge: Path | None = None) -> Path:
    """Ensure final_model + retrofit_sim are importable. Returns challenge root."""
    root = challenge or challenge_dir()
    os.environ.setdefault("QUAVIZ_CHALLENGE_DIR", str(root))
    final = final_model_dir(root)
    sim_src = root / "src"
    for path in (final, sim_src):
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)
    return root


def repo_dotenv() -> Path:
    return REPO_ROOT / ".env"
