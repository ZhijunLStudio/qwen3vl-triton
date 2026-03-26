"""
ui/tui.py - Rich TUI for real-time display

Shows: metrics panel, iteration history, live agent log.
Falls back to plain text if Rich is not installed.
"""

import threading
from collections import deque
from datetime import datetime
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import GOALS
from core.metrics import ScoreEngine
from core.state import State

try:
    from rich.console import Console
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# ─── Log entry ───────────────────────────────────────────────────────────────

LEVEL_COLORS = {
    "info":    "white",
    "tool":    "cyan",
    "result":  "green",
    "error":   "red",
    "think":   "yellow",
    "git":     "magenta",
    "bench":   "blue",
    "memory":  "bright_black",
}


class TUI:
    """
    Terminal UI backed by Rich Live.

    Usage:
        tui = TUI()
        tui.start(state)
        tui.log("message", "info")
        tui.update(state)   # refresh display
        tui.stop()
    """

    def __init__(self, max_log_lines: int = 300):
        self._logs: deque = deque(maxlen=max_log_lines)
        self._lock   = threading.Lock()
        self._live: Optional[Live] = None
        self._state: Optional[State] = None
        self._console = Console() if HAS_RICH else None

    # ── Public API ───────────────────────────────────────────────────────────

    def log(self, msg: str, level: str = "info"):
        ts = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            self._logs.append({"ts": ts, "msg": msg, "level": level})

        if not HAS_RICH or self._live is None:
            # Plain text fallback
            print(f"[{ts}] {msg}")

    def start(self, state: State):
        self._state = state
        if HAS_RICH:
            self._live = Live(
                self._render(state),
                refresh_per_second=2,
                screen=False,
                console=self._console,
            )
            self._live.start()

    def update(self, state: State):
        self._state = state
        if HAS_RICH and self._live:
            self._live.update(self._render(state))

    def stop(self):
        if HAS_RICH and self._live:
            self._live.stop()
            self._live = None

    # ── Rendering ────────────────────────────────────────────────────────────

    def _render(self, state: State) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
        )
        layout["body"].split_row(
            Layout(name="left",  ratio=1, minimum_size=22),
            Layout(name="right", ratio=3),
        )
        layout["left"].split_column(
            Layout(name="metrics", ratio=3),
            Layout(name="history", ratio=2),
        )

        s    = state.data
        best = s.get("best", {})

        # ── Header ──────────────────────────────────────────────────────────
        iter_n  = s.get("iteration", 0)
        best_sc = best.get("score", 0.0)
        status  = "✅ DONE" if s.get("goals_achieved") else "🔄 Running"
        improv  = s.get("improvements", 0)
        layout["header"].update(Panel(
            f"  🚀 AICAS OptimAgent  │  Iter #{iter_n}  │  "
            f"Best Score: {best_sc:.4f}  │  Improvements: {improv}  │  {status}",
            style="bold blue",
        ))

        # ── Metrics panel ────────────────────────────────────────────────────
        tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 1), expand=True)
        tbl.add_column("k", style="dim",  no_wrap=True)
        tbl.add_column("v", style="bold", no_wrap=True)

        def mrow(label: str, m: dict, key: str) -> tuple:
            v = m.get(key, None)
            if v is None:
                return label, "[dim]N/A[/dim]"
            if key == "ttft_ms":
                g = GOALS["ttft_ms"]
                c = "green" if v <= g else ("yellow" if v <= g * 1.5 else "red")
                return label, f"[{c}]{v:.1f}ms[/{c}]"
            elif key == "throughput":
                g = GOALS["throughput"]
                c = "green" if v >= g else ("yellow" if v >= g * 0.5 else "red")
                return label, f"[{c}]{v:.1f} t/s[/{c}]"
            elif key == "accuracy":
                c = "green" if v >= 0.30 else "yellow"
                return label, f"[{c}]{v:.1%}[/{c}]"
            elif key == "score":
                c = "green" if v > 1.0 else ("yellow" if v > 0.5 else "white")
                return label, f"[{c}]{v:.4f}[/{c}]"
            return label, str(v)

        for section, data in [
            ("── Baseline", s.get("baseline", {})),
            ("── Current",  s.get("current",  {})),
            ("── Best",     s.get("best",     {})),
        ]:
            tbl.add_row(f"[dim]{section}[/dim]", "")
            for lbl, key in [("TTFT", "ttft_ms"), ("TP", "throughput"),
                              ("Acc", "accuracy"), ("Score", "score")]:
                k, v = mrow(lbl, data, key)
                tbl.add_row(k, v)

        tbl.add_row("", "")
        cf = s.get("consec_fails", 0)
        tbl.add_row("Improv", f"[green]{improv}[/green]")
        tbl.add_row(
            "C.Fail",
            f"[red]{cf}[/red]" if cf >= 3 else f"[yellow]{cf}[/yellow]" if cf else "0"
        )
        tbl.add_row("Best git", s.get("best_git_hash", "N/A")[:8])

        layout["metrics"].update(Panel(tbl, title="📊 Metrics", border_style="blue"))

        # ── History panel ────────────────────────────────────────────────────
        hot   = s.get("hot_history", [])
        lines = []
        for r in hot[-8:]:
            icon = "✅" if r.get("outcome") == "improved" else "❌"
            d    = r.get("delta", 0.0)
            act  = r.get("action", "")[:22]
            lines.append(f"{icon} #{r.get('iter','?')}  {d:+.3f}  {act}")

        hist_txt = "\n".join(lines) if lines else "(no history)"
        layout["history"].update(Panel(hist_txt, title="📋 History", border_style="dim"))

        # ── Log panel ────────────────────────────────────────────────────────
        with self._lock:
            recent = list(self._logs)[-40:]

        log_lines = []
        for e in recent:
            lv    = e.get("level", "info")
            color = LEVEL_COLORS.get(lv, "white")
            log_lines.append(f"[dim]{e['ts']}[/dim] [{color}]{e['msg']}[/{color}]")

        log_txt = "\n".join(log_lines) if log_lines else "[dim]Waiting for agent...[/dim]"
        layout["right"].update(Panel(
            log_txt,
            title="🤖 Agent Log",
            border_style="green",
        ))

        return layout

    # ── Log callback factory ─────────────────────────────────────────────────

    def make_log_cb(self, state: State):
        """Returns a callback (msg, level) that logs and refreshes the display."""
        def cb(msg: str, level: str = "info"):
            self.log(msg, level)
            self.update(state)
        return cb
