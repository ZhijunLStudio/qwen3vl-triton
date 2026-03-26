"""
core/metrics.py - Score computation engine (Python only, never delegated to LLM)

All scoring logic lives here. The LLM sees the result but never computes it.
"""

from typing import Tuple, Dict
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import GOALS, SCORE_WEIGHTS


class ScoreEngine:
    """
    Competition score calculator.

    Formula (matches SKILL.md exactly):
        score = accuracy × W_acc
              + ttft_improvement × W_ttft
              + throughput_improvement × W_tp

    where:
        accuracy               = current hit rate (e.g. 0.35 for 7/20 samples)
        ttft_improvement       = (baseline_ttft - current_ttft) / baseline_ttft
        throughput_improvement = (current_tp - baseline_tp) / baseline_tp

    At baseline:  score = baseline_accuracy × 0.40   (both improvement terms = 0)
    Example:      acc=0.35, TTFT=30ms(↓from 65), TP=400(↑from 65)
                  = 0.35×0.4 + 0.538×0.3 + 5.15×0.3 = 1.846

    Accuracy constraint (checked separately): current_acc / baseline_acc >= 0.95
    """

    @staticmethod
    def compute(metrics: Dict, baseline: Dict) -> float:
        """
        Compute competition score.
        Score increases as TTFT drops, throughput rises, accuracy stays high.
        """
        if not baseline or not metrics:
            return 0.0

        b_ttft = baseline.get("ttft_ms",   float("inf"))
        b_tp   = max(baseline.get("throughput", 1.0), 1e-9)

        c_acc  = metrics.get("accuracy",   0.0)
        c_ttft = metrics.get("ttft_ms",    float("inf"))
        c_tp   = metrics.get("throughput", 0.0)

        if b_ttft > 0 and c_ttft != float("inf"):
            ttft_imp = (b_ttft - c_ttft) / b_ttft
        else:
            ttft_imp = 0.0

        tp_imp = (c_tp - b_tp) / b_tp

        W = SCORE_WEIGHTS
        score = (
            c_acc    * W["accuracy"] +
            ttft_imp * W["ttft"]     +
            tp_imp   * W["throughput"]
        )
        return round(score, 6)

    @staticmethod
    def is_accuracy_valid(metrics: Dict, baseline: Dict) -> Tuple[bool, str]:
        """
        Returns (valid, reason).
        Accuracy must not drop below GOALS['acc_ratio'] of baseline.
        """
        b_acc = max(baseline.get("accuracy", 1.0), 1e-9)
        c_acc = metrics.get("accuracy", 0.0)
        ratio = c_acc / b_acc

        if ratio < GOALS["acc_ratio"]:
            return False, (
                f"Accuracy dropped to {ratio:.1%} of baseline "
                f"(minimum allowed: {GOALS['acc_ratio']:.0%})"
            )
        return True, "ok"

    @staticmethod
    def goals_met(metrics: Dict) -> bool:
        """Returns True if both TTFT and throughput goals are reached."""
        return (
            metrics.get("ttft_ms",    float("inf")) <= GOALS["ttft_ms"] and
            metrics.get("throughput", 0.0)           >= GOALS["throughput"]
        )

    @staticmethod
    def delta(new: Dict, old: Dict, baseline: Dict) -> float:
        """Score delta between new and old metrics."""
        return ScoreEngine.compute(new, baseline) - ScoreEngine.compute(old, baseline)

    @staticmethod
    def fmt(m: Dict) -> str:
        """Human-readable metrics string."""
        ttft = m.get("ttft_ms",    float("inf"))
        tp   = m.get("throughput", 0.0)
        acc  = m.get("accuracy",   0.0)
        sc   = m.get("score",      0.0)
        return (
            f"TTFT={ttft:.1f}ms  "
            f"TP={tp:.1f}t/s  "
            f"Acc={acc:.1%}  "
            f"Score={sc:.4f}"
        )

    @staticmethod
    def fmt_delta(delta: float) -> str:
        sign = "+" if delta >= 0 else ""
        return f"Δscore={sign}{delta:.4f}"
