"""tests/test_memory.py - Unit tests for Memory manager"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pathlib import Path
from memory.manager import Memory


def make_mem():
    tmp = Path(tempfile.mktemp(suffix=".json"))
    return Memory(path=tmp), tmp


def test_add_and_retrieve():
    mem, tmp = make_mem()
    mem.add("worked",      "TF32 improved throughput by 10%")
    mem.add("failed",      "torch.compile caused OOM")
    mem.add("observation", "Vision encoder takes 40% of TTFT")
    mem.add("strategy",    "Quantize language model first")

    prompt = mem.to_prompt_str()
    assert "TF32" in prompt
    assert "torch.compile" in prompt
    assert "Vision encoder" in prompt
    assert "Quantize" in prompt
    tmp.unlink(missing_ok=True)
    print("  ✓ add/retrieve works")


def test_persistence():
    tmp = Path(tempfile.mktemp(suffix=".json"))
    m1 = Memory(path=tmp)
    m1.add("worked", "Flash Attention 2 reduced TTFT by 15ms")

    m2 = Memory(path=tmp)
    assert "Flash Attention" in m2.to_prompt_str()
    tmp.unlink(missing_ok=True)
    print("  ✓ persistence works")


def test_max_per_category():
    mem, tmp = make_mem()
    for i in range(30):
        mem.add("failed", f"attempt {i} failed")
    count = mem.count()["failed"]
    from config import CONTEXT
    max_expected = CONTEXT["max_lessons"] // len(Memory.CATEGORIES)
    assert count <= max_expected, f"Expected <= {max_expected}, got {count}"
    tmp.unlink(missing_ok=True)
    print(f"  ✓ max_per_category capping works (kept {count}, limit {max_expected})")


def test_invalid_category():
    mem, tmp = make_mem()
    try:
        mem.add("invalid_cat", "something")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    tmp.unlink(missing_ok=True)
    print("  ✓ invalid category raises ValueError")


def test_prompt_str_truncation():
    mem, tmp = make_mem()
    for _ in range(30):
        mem.add("observation", "x" * 200)  # long text
    prompt = mem.to_prompt_str(max_chars=500)
    assert len(prompt) <= 500, f"Should truncate to 500, got {len(prompt)}"
    tmp.unlink(missing_ok=True)
    print("  ✓ to_prompt_str truncation works")


def test_clear():
    mem, tmp = make_mem()
    mem.add("worked", "something")
    mem.clear()
    assert mem.count() == {"worked": 0, "failed": 0, "observation": 0, "strategy": 0}
    tmp.unlink(missing_ok=True)
    print("  ✓ clear works")


if __name__ == "__main__":
    print("=== test_memory ===")
    test_add_and_retrieve()
    test_persistence()
    test_max_per_category()
    test_invalid_category()
    test_prompt_str_truncation()
    test_clear()
    print("All tests passed ✓")
