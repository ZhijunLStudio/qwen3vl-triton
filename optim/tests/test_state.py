"""tests/test_state.py - Unit tests for State management"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pathlib import Path
from core.state import State


def make_state():
    tmp = Path(tempfile.mktemp(suffix=".json"))
    return State(path=tmp), tmp


def test_defaults():
    state, tmp = make_state()
    assert state.get("iteration") == 0
    assert state.get("improvements") == 0
    assert state.get("baseline") == {}
    tmp.unlink(missing_ok=True)
    print("  ✓ defaults correct")


def test_set_get():
    state, tmp = make_state()
    state.set("iteration", 42)
    assert state.get("iteration") == 42
    tmp.unlink(missing_ok=True)
    print("  ✓ set/get works")


def test_persistence():
    tmp = Path(tempfile.mktemp(suffix=".json"))
    s1 = State(path=tmp)
    s1.set("best", {"ttft_ms": 30.0, "score": 1.5})
    s1.increment("improvements")

    # Load fresh from disk
    s2 = State(path=tmp)
    assert s2.get("best") == {"ttft_ms": 30.0, "score": 1.5}
    assert s2.get("improvements") == 1
    tmp.unlink(missing_ok=True)
    print("  ✓ persistence works")


def test_update():
    state, tmp = make_state()
    state.update({"iteration": 5, "improvements": 3})
    assert state.get("iteration") == 5
    assert state.get("improvements") == 3
    tmp.unlink(missing_ok=True)
    print("  ✓ update works")


def test_append_hot():
    state, tmp = make_state()
    for i in range(60):
        state.append_hot({"iter": i, "outcome": "improved"})
    hot = state.get("hot_history")
    assert len(hot) == 50, f"hot_history should cap at 50, got {len(hot)}"
    tmp.unlink(missing_ok=True)
    print("  ✓ hot_history capped at 50")


def test_reset_baseline():
    state, tmp = make_state()
    state.set("baseline", {"ttft_ms": 65.0})
    state.set("best", {"score": 1.5})
    state.reset_baseline()
    assert state.get("baseline") == {}
    assert state.get("best") == {}
    tmp.unlink(missing_ok=True)
    print("  ✓ reset_baseline works")


if __name__ == "__main__":
    print("=== test_state ===")
    test_defaults()
    test_set_get()
    test_persistence()
    test_update()
    test_append_hot()
    test_reset_baseline()
    print("All tests passed ✓")
