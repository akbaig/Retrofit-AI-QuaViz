"""Lightweight backend unit checks (no model load)."""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from quaviz_backend.paths import (  # noqa: E402
    REPO_ROOT,
    bootstrap_imports,
    challenge_dir,
    data_dir,
    final_model_dir,
    model_path,
    policy_path,
)


def test_default_layout():
    root = challenge_dir()
    assert root.name == "retrofit_starter"
    assert (root / "src" / "retrofit_sim").is_dir()
    assert model_path().name == "model.joblib"
    assert policy_path().name == "policy.json"
    assert data_dir().name == "full"
    assert final_model_dir().is_dir()
    assert REPO_ROOT.name == "Retrofit-AI-QuaViz"


def test_bootstrap_imports_deploy():
    bootstrap_imports()
    from deploy.aggregate import asset_id_from_topic

    assert asset_id_from_topic("retrofit/site_a/PUMP-08/telemetry") == "PUMP-08"
    assert asset_id_from_topic("other") is None
