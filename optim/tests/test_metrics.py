"""tests/test_metrics.py - Unit tests for ScoreEngine"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.metrics import ScoreEngine

BASELINE = {"ttft_ms": 65.0, "throughput": 65.0, "accuracy": 0.35}


def test_baseline_score():
    """
    At baseline, ttft_imp=0, tp_imp=0, so:
    score = 0.35 * 0.4 + 0 + 0 = 0.14
    """
    score = ScoreEngine.compute(BASELINE, BASELINE)
    expected = round(0.35 * 0.4, 6)
    assert abs(score - expected) < 1e-5, f"Expected {expected}, got {score}"
    print(f"  ✓ baseline score = {score:.6f} (expected {expected:.6f})")


def test_better_ttft_increases_score():
    """Better TTFT → score > baseline score."""
    improved = {"ttft_ms": 30.0, "throughput": 65.0, "accuracy": 0.35}
    baseline_score = ScoreEngine.compute(BASELINE, BASELINE)
    new_score      = ScoreEngine.compute(improved, BASELINE)
    assert new_score > baseline_score, f"Better TTFT should increase score: {new_score} vs {baseline_score}"
    ttft_imp = (65.0 - 30.0) / 65.0
    expected = round(0.35 * 0.4 + ttft_imp * 0.3, 6)
    assert abs(new_score - expected) < 1e-5, f"Expected {expected}, got {new_score}"
    print(f"  ✓ better TTFT score = {new_score:.4f}  (Δ={new_score-baseline_score:+.4f})")


def test_better_throughput_increases_score():
    """Better throughput → score > baseline score."""
    improved = {"ttft_ms": 65.0, "throughput": 400.0, "accuracy": 0.35}
    baseline_score = ScoreEngine.compute(BASELINE, BASELINE)
    new_score      = ScoreEngine.compute(improved, BASELINE)
    assert new_score > baseline_score, f"Better TP should increase score: {new_score} vs {baseline_score}"
    tp_imp = (400.0 - 65.0) / 65.0
    expected = round(0.35 * 0.4 + tp_imp * 0.3, 6)
    assert abs(new_score - expected) < 1e-5, f"Expected {expected}, got {new_score}"
    print(f"  ✓ better TP score = {new_score:.4f}  (Δ={new_score-baseline_score:+.4f})")


def test_full_optimization_score():
    """TTFT=30ms, TP=400, acc=0.35 → concrete expected value."""
    optimized = {"ttft_ms": 30.0, "throughput": 400.0, "accuracy": 0.35}
    score = ScoreEngine.compute(optimized, BASELINE)
    ttft_imp = (65.0 - 30.0) / 65.0
    tp_imp   = (400.0 - 65.0) / 65.0
    expected = round(0.35 * 0.4 + ttft_imp * 0.3 + tp_imp * 0.3, 6)
    assert abs(score - expected) < 1e-5, f"Expected {expected}, got {score}"
    print(f"  ✓ full optimization score = {score:.4f}")


def test_accuracy_drop_reduces_score():
    """Same speed but lower accuracy → lower score."""
    degraded = {"ttft_ms": 30.0, "throughput": 400.0, "accuracy": 0.10}
    full     = {"ttft_ms": 30.0, "throughput": 400.0, "accuracy": 0.35}
    score_full     = ScoreEngine.compute(full, BASELINE)
    score_degraded = ScoreEngine.compute(degraded, BASELINE)
    assert score_degraded < score_full, "Lower accuracy should reduce score"
    print(f"  ✓ accuracy drop: {score_full:.4f} → {score_degraded:.4f}")


def test_accuracy_valid():
    """Accuracy must stay >= 95% of baseline."""
    ok, _       = ScoreEngine.is_accuracy_valid({"accuracy": 0.35}, BASELINE)
    assert ok

    ok2, reason = ScoreEngine.is_accuracy_valid({"accuracy": 0.10}, BASELINE)
    assert not ok2, f"Should fail: {reason}"
    print(f"  ✓ accuracy_valid: rejection reason: {reason}")


def test_accuracy_valid_borderline():
    """Exactly at 95% threshold should pass."""
    min_acc = 0.35 * 0.95  # = 0.3325
    ok, _   = ScoreEngine.is_accuracy_valid({"accuracy": min_acc}, BASELINE)
    assert ok, "Exactly at threshold should be valid"
    ok2, _ = ScoreEngine.is_accuracy_valid({"accuracy": min_acc - 0.01}, BASELINE)
    assert not ok2, "Just below threshold should fail"
    print(f"  ✓ accuracy_valid borderline works (threshold={min_acc:.4f})")


def test_goals_met():
    assert ScoreEngine.goals_met({"ttft_ms": 25.0, "throughput": 450.0})
    assert not ScoreEngine.goals_met({"ttft_ms": 35.0, "throughput": 450.0})
    assert not ScoreEngine.goals_met({"ttft_ms": 25.0, "throughput": 300.0})
    print("  ✓ goals_met works")


def test_delta():
    improved = {"ttft_ms": 30.0, "throughput": 400.0, "accuracy": 0.35}
    d = ScoreEngine.delta(improved, BASELINE, BASELINE)
    assert d > 0, f"Improvement delta should be positive, got {d}"
    print(f"  ✓ delta = {d:+.4f}")


def test_fmt():
    m = {"ttft_ms": 65.47, "throughput": 65.43, "accuracy": 0.35, "score": 0.14}
    s = ScoreEngine.fmt(m)
    assert "65.5ms" in s or "65.4ms" in s
    assert "35.0%" in s
    print(f"  ✓ fmt: {s}")


def test_empty_inputs():
    assert ScoreEngine.compute({}, BASELINE) == 0.0
    assert ScoreEngine.compute(BASELINE, {}) == 0.0
    print("  ✓ empty inputs handled")


if __name__ == "__main__":
    print("=== test_metrics ===")
    test_baseline_score()
    test_better_ttft_increases_score()
    test_better_throughput_increases_score()
    test_full_optimization_score()
    test_accuracy_drop_reduces_score()
    test_accuracy_valid()
    test_accuracy_valid_borderline()
    test_goals_met()
    test_delta()
    test_fmt()
    test_empty_inputs()
    print("All tests passed ✓")
