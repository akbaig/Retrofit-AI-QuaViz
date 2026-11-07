"""Reproduction contracts without refitting the frozen sensor models."""
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_sensor03_defaults_preserve_training_order_and_parameters():
    from train_sensor import build_parser, expert_parameters

    args = build_parser().parse_args(["--name", "reproduction"])
    assert args.datasets == ["original_train", "train", "train_balanced",
                             "train_crossed", "train_sensor_crossed"]
    assert args.reference == "hist_01"
    assert args.evaluate == ["development", "original_validation"]
    assert expert_parameters(args) == {
        "seed": 20261107, "max_positive_per_type_channel": 1800,
        "max_negative_per_channel": 3500, "max_iter": 140, "routed": True,
    }


def test_retention_preserves_source_order_and_drops_identity_features():
    from train_sensor import retained_columns

    columns = ["snapshot__rms", "residual__motor_current_a", "asset_id",
               "phys__electrical_power_factor", "ctx__ambient_temp_c",
               "fresh__motor_current_a", "level__motor_current_a",
               "phys__head_curve_residual", "rating__power", "obs__gap",
               "equal_updates__motor_current_a", "count__motor_current_a"]
    frame = pd.DataFrame(columns=columns)
    assert retained_columns(frame) == [columns[i] for i in (1, 3, 4, 5, 6, 7, 9, 10, 11)]


def test_existing_run_is_preserved_before_any_data_or_model_load(tmp_path, monkeypatch):
    import train_sensor

    monkeypatch.setattr(train_sensor, "HERE", tmp_path)
    output = tmp_path / "runs" / "keep"
    output.mkdir(parents=True)
    model = output / "model.joblib"
    model.write_bytes(b"existing model")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        train_sensor.main(["--name", "keep"])
    assert model.read_bytes() == b"existing model"
    assert list(output.iterdir()) == [model]


@pytest.mark.parametrize("flag", ["--datasets", "--evaluate"])
def test_holdout_is_not_a_supported_cli_input(flag):
    from train_sensor import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--name", "reproduction", flag, "holdout"])
