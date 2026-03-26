"""
core/state.py - Persistent agent state management

Saves/loads from .optim_state.json. Thread-safe.
Tracks: baseline, current, best metrics; iteration history; git hashes.
"""

import json
import threading
from pathlib import Path
from typing import Any, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import STATE_FILE


_DEFAULTS = {
    "iteration":       0,
    "total_attempts":  0,
    "improvements":    0,
    "consec_fails":    0,
    "baseline":        {},
    "current":         {},
    "best":            {},
    "best_git_hash":   "",
    "hot_history":     [],   # last N full iteration records (hot memory)
    "warm_summary":    "",   # compressed older history (warm memory)
    "goals_achieved":  False,
    "last_bench":      None, # result of most recent run_benchmark call
}


class State:
    """
    Thread-safe persistent state.

    Usage:
        state = State()
        state.set("iteration", 5)
        val = state.get("iteration")       # 5
        state.update({"a": 1, "b": 2})
    """

    def __init__(self, path: Optional[Path] = None):
        self._path = path or STATE_FILE
        self._lock = threading.Lock()
        self._data = self._load()

    # ── Persistence ─────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if self._path.exists():
            try:
                loaded = json.loads(self._path.read_text(encoding="utf-8"))
                # Merge with defaults so new keys appear automatically
                merged = dict(_DEFAULTS)
                merged.update(loaded)
                return merged
            except Exception:
                pass
        return dict(_DEFAULTS)

    def save(self):
        with self._lock:
            data_copy = dict(self._data)
        self._path.write_text(
            json.dumps(data_copy, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    # ── Access ───────────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any):
        with self._lock:
            self._data[key] = value
        self.save()

    def update(self, d: dict):
        with self._lock:
            self._data.update(d)
        self.save()

    def increment(self, key: str, by: int = 1):
        with self._lock:
            self._data[key] = self._data.get(key, 0) + by
        self.save()

    @property
    def data(self) -> dict:
        """Return a shallow copy of state dict."""
        with self._lock:
            return dict(self._data)

    def reset(self):
        """Reset to defaults (keeps file path)."""
        with self._lock:
            self._data = dict(_DEFAULTS)
        self.save()

    def reset_baseline(self):
        """Clear baseline/current/best so they will be re-measured."""
        self.update({
            "baseline": {},
            "current":  {},
            "best":     {},
            "best_git_hash": "",
            "last_bench": None,
        })

    # ── Hot History Helpers ──────────────────────────────────────────────────

    def append_hot(self, record: dict):
        """Add one iteration record to hot history (keeps last 50)."""
        with self._lock:
            hot = self._data.get("hot_history", [])
            hot.append(record)
            self._data["hot_history"] = hot[-50:]
        self.save()

    def __repr__(self):
        s = self._data
        return (
            f"<State iter={s.get('iteration')} "
            f"improvements={s.get('improvements')} "
            f"consec_fails={s.get('consec_fails')}>"
        )
