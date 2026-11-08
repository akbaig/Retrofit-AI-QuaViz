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
    _emitted: set = field(default_factory=set)

    def push(self, payload: dict) -> list:
        """Add one sample. Returns list of newly completed 1-min feature frames (0 or 1)."""
        row = normalize_row(payload, self.asset_id)
        minute = row["timestamp_utc"].floor("min")
        # Ignore late samples for minutes already closed (prevents duplicate aggregates).
        if minute in self._emitted:
            return []
        done = []
        if self._current_minute is None:
            self._current_minute = minute
        elif minute != self._current_minute:
            done.extend(self._flush())
            # After flush, skip if the new minute was already emitted earlier.
            if minute in self._emitted:
                self._current_minute = None
                self._rows = []
                return done
            self._current_minute = minute
        self._rows.append(row)
        return done

    def flush(self) -> list:
        """Flush the open minute (EOF / end of stream)."""
        return self._flush()

    def _flush(self) -> list:
        if not self._rows:
            self._current_minute = None
            return []
        minute = self._current_minute
        feats = rows_to_features(self._rows)
        self._rows = []
        self._current_minute = None
        if minute is not None:
            if minute in self._emitted:
                return []
            self._emitted.add(minute)
        if len(feats):
            self.completed_features.append(feats)
            return [feats]
        return []

    def features_table(self) -> "pd.DataFrame":
        if not self.completed_features:
            return pd.DataFrame()
        return pd.concat(self.completed_features, ignore_index=True)
