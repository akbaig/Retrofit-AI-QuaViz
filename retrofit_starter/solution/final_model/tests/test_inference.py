import pandas as pd
from infer import observed_horizon
from test_causality import toy


def test_truncated_input_does_not_advance_clock_to_declared_window_end():
    f,a,c=toy(360)
    cutoff=f.window_start_utc.iloc[100]
    horizon=observed_horizon(f.iloc[:100],a,c.iloc[:100],None)
    assert horizon==cutoff
    assert horizon<pd.Timestamp(a.window_end_utc)


def test_independent_context_clock_can_reveal_gateway_silence():
    f,a,c=toy(360)
    horizon=observed_horizon(f.iloc[:80],a,c.iloc[:101],None)
    assert horizon==c.timestamp_utc.iloc[100]


def test_unavailable_snapshot_is_not_used_to_advance_time():
    f,a,c=toy(10)
    # A snapshot record is available only when the full acquisition completes.
    s=pd.DataFrame(dict(asset_id=['TOY'],timestamp_utc=[f.window_start_utc.iloc[8]],
        n_samples=[10240],sample_rate_hz=[10240]))
    assert observed_horizon(f.iloc[:8],a,c.iloc[:8],s)==f.window_start_utc.iloc[8]+pd.Timedelta(seconds=1)
