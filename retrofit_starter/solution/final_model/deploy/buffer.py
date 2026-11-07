"""Minute-boundary buffer for 1 s telemetry rows."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .aggregate import normalize_row, rows_to_features


@dataclass
class MinuteBuffer:
    asset_id: str
    _current_minute: pd.Timestamp | None = None
    _rows: list[dict] = field(default_factory=list)
    completed_features: list = field(default_factory=list)

    def push(self, payload: dict) -> list:
        """Add one sample. Returns list of newly completed 1-min feature frames (0 or 1)."""
        row = normalize_row(payload, self.asset_id)
        minute = row["timestamp_utc"].floor("min")
        done = []
        if self._current_minute is None:
            self._current_minute = minute
        elif minute != self._current_minute:
            done.extend(self._flush())
            self._current_minute = minute
        self._rows.append(row)
        return done

    def flush(self) -> list:
        """Flush the open minute (EOF / end of stream)."""
        return self._flush()

    def _flush(self) -> list:
        if not self._rows:
            return []
        feats = rows_to_features(self._rows)
        self._rows = []
        self._current_minute = None
        if len(feats):
            self.completed_features.append(feats)
            return [feats]
        return []

    def features_table(self) -> "pd.DataFrame":
        import pandas as pd

        if not self.completed_features:
            return pd.DataFrame()
        return pd.concat(self.completed_features, ignore_index=True)
