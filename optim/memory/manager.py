"""
memory/manager.py - Three-tier memory management

Tier 1 (Hot):     Last N full iteration records (in state.hot_history)
Tier 2 (Warm):    Compressed older iterations (in state.warm_summary)
Tier 3 (Cold):    Persistent lessons by category (in .optim_memory.json)
"""

import json
import threading
from pathlib import Path
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import MEMORY_FILE, CONTEXT


class Memory:
    """
    Persistent long-term memory (Cold tier).
    Stores lessons by category: worked / failed / observation / strategy.
    """

    CATEGORIES = ("worked", "failed", "observation", "strategy")

    def __init__(self, path: Optional[Path] = None):
        self._path = path or MEMORY_FILE
        self._lock = threading.Lock()
        self._data = self._load()

    # ── Persistence ─────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {cat: [] for cat in self.CATEGORIES}

    def save(self):
        with self._lock:
            data_copy = dict(self._data)
        self._path.write_text(
            json.dumps(data_copy, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def add(self, category: str, content: str):
        """Add a lesson. Old entries are trimmed if over limit."""
        if category not in self.CATEGORIES:
            raise ValueError(f"Invalid category '{category}'. Must be one of {self.CATEGORIES}")
        from datetime import datetime
        entry = {
            "ts":   datetime.now().strftime("%m-%d %H:%M"),
            "text": content.strip(),
        }
        with self._lock:
            if category not in self._data:
                self._data[category] = []
            self._data[category].append(entry)
            # Keep last N per category
            max_per_cat = CONTEXT["max_lessons"] // len(self.CATEGORIES)
            self._data[category] = self._data[category][-max_per_cat:]
        self.save()

    def get_all(self) -> dict:
        with self._lock:
            return {k: list(v) for k, v in self._data.items()}

    def clear(self):
        with self._lock:
            self._data = {cat: [] for cat in self.CATEGORIES}
        self.save()

    # ── Serialization for prompt ─────────────────────────────────────────────

    def to_prompt_str(self, max_chars: int = 3000) -> str:
        """
        Format all lessons as a compact string for inclusion in LLM prompts.
        Returns at most max_chars characters.
        """
        ICONS = {
            "worked":      "[WORKED]",
            "failed":      "[FAILED]",
            "observation": "[OBS]",
            "strategy":    "[STRATEGY]",
        }
        lines = []
        with self._lock:
            for cat in self.CATEGORIES:
                items = self._data.get(cat, [])
                for item in items[-8:]:  # show last 8 per category
                    lines.append(f"{ICONS[cat]} {item['text']}")

        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[-max_chars:]
        return text or "(no lessons recorded yet)"

    def count(self) -> dict:
        with self._lock:
            return {k: len(v) for k, v in self._data.items()}

    def __repr__(self):
        c = self.count()
        return f"<Memory {c}>"
