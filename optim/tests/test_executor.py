"""tests/test_executor.py - Unit tests for ToolExecutor (no GPU/model needed)"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pathlib import Path
from core.state import State
from memory.manager import Memory
from core.executor import ToolExecutor


def make_executor():
    tmp_state  = Path(tempfile.mktemp(suffix="_state.json"))
    tmp_memory = Path(tempfile.mktemp(suffix="_mem.json"))
    state  = State(path=tmp_state)
    memory = Memory(path=tmp_memory)
    logs   = []
    ex     = ToolExecutor(state, memory, log_cb=lambda m, _="info": logs.append(m))
    # returns: ex, state, memory, logs, tmp_state, tmp_memory
    return ex, state, memory, logs, tmp_state, tmp_memory


def cleanup(*paths):
    for p in paths:
        Path(p).unlink(missing_ok=True)


# ── read ──────────────────────────────────────────────────────────────────────

def test_read_existing_file():
    ex, _, _, _, *tmp = make_executor()
    result = ex.execute("read", {"path": "evaluation_wrapper.py"})
    assert "ERROR" not in result, f"Should read evaluation_wrapper.py: {result[:200]}"
    assert "VLMModel" in result
    cleanup(*tmp)
    print("  ✓ read(evaluation_wrapper.py) works")


def test_read_missing_file():
    ex, *rest = make_executor()
    result = ex.execute("read", {"path": "nonexistent_xyz.py"})
    assert "ERROR" in result
    cleanup(*rest[3:])
    print("  ✓ read(missing) returns ERROR")


def test_read_with_limit():
    ex, *rest = make_executor()
    # benchmark.py lives directly in WORK_DIR
    result = ex.execute("read", {"path": "benchmark.py", "limit": 5})
    lines = result.strip().splitlines()
    assert len(lines) <= 5, f"Expected <= 5 lines, got {len(lines)}"
    cleanup(*rest[3:])
    print(f"  ✓ read with limit=5 returns {len(lines)} lines")


# ── write ─────────────────────────────────────────────────────────────────────

def test_write_wrong_file():
    ex, *rest = make_executor()
    result = ex.execute("write", {"path": "config.py", "content": "x"})
    assert "ERROR" in result
    cleanup(*rest[3:])
    print("  ✓ write to non-wrapper file blocked")


# ── bash ──────────────────────────────────────────────────────────────────────

def test_bash_echo():
    ex, *rest = make_executor()
    result = ex.execute("bash", {"command": "echo hello_world"})
    assert "hello_world" in result, f"Expected 'hello_world' in: {result}"
    cleanup(*rest[3:])
    print("  ✓ bash(echo) works")


def test_bash_python():
    ex, *rest = make_executor()
    result = ex.execute("bash", {"command": "python3 -c 'print(1+1)'"})
    assert "2" in result, f"Expected '2' in: {result}"
    cleanup(*rest[3:])
    print("  ✓ bash(python3) works")


def test_bash_timeout():
    ex, *rest = make_executor()
    result = ex.execute("bash", {"command": "sleep 10", "timeout": 2})
    assert "TIMEOUT" in result or "timeout" in result.lower()
    cleanup(*rest[3:])
    print("  ✓ bash timeout works")


# ── grep ──────────────────────────────────────────────────────────────────────

def test_grep_finds_pattern():
    ex, *rest = make_executor()
    result = ex.execute("grep", {"pattern": "VLMModel", "path": ".", "glob": "*.py"})
    assert "VLMModel" in result, f"Expected to find VLMModel: {result[:200]}"
    cleanup(*rest[3:])
    print("  ✓ grep(VLMModel) finds matches")


def test_grep_no_match():
    ex, *rest = make_executor()
    # Pattern that cannot appear in source: null bytes in a grep pattern
    result = ex.execute("grep", {"pattern": "QQNOEXIST_9z9z9z_QQNOEXIST", "path": ".", "glob": "*.json"})
    assert "no matches" in result.lower() or result.strip() == "(no matches)" or not result.strip()
    cleanup(*rest[3:])
    print("  ✓ grep no-match returns empty")


# ── glob ──────────────────────────────────────────────────────────────────────

def test_glob_py_files():
    ex, *rest = make_executor()
    result = ex.execute("glob", {"pattern": "*.py"})
    assert ".py" in result
    cleanup(*rest[3:])
    print("  ✓ glob(*.py) finds files")


# ── add_lesson ────────────────────────────────────────────────────────────────

def test_add_lesson():
    ex, _, memory, _, *tmp = make_executor()
    result = ex.execute("add_lesson", {"category": "worked", "content": "TF32 helps a lot"})
    assert "OK" in result
    assert "TF32" in memory.to_prompt_str()
    cleanup(*tmp)
    print("  ✓ add_lesson works")


def test_add_lesson_invalid_category():
    ex, *rest = make_executor()
    result = ex.execute("add_lesson", {"category": "badcat", "content": "test"})
    assert "ERROR" in result
    cleanup(*rest[3:])
    print("  ✓ add_lesson invalid category blocked")


# ── get_status ────────────────────────────────────────────────────────────────

def test_get_status():
    import json
    ex, state, _, _, *tmp = make_executor()
    state.set("iteration", 7)
    result = ex.execute("get_status", {})
    data = json.loads(result)
    assert data["iteration"] == 7
    cleanup(*tmp)
    print("  ✓ get_status returns correct iteration")


# ── unknown tool ──────────────────────────────────────────────────────────────

def test_unknown_tool():
    ex, *rest = make_executor()
    result = ex.execute("nonexistent_tool_xyz", {})
    assert "ERROR" in result
    cleanup(*rest[3:])
    print("  ✓ unknown tool returns ERROR")


if __name__ == "__main__":
    print("=== test_executor ===")
    test_read_existing_file()
    test_read_missing_file()
    test_read_with_limit()
    test_write_wrong_file()
    test_bash_echo()
    test_bash_python()
    test_bash_timeout()
    test_grep_finds_pattern()
    test_grep_no_match()
    test_glob_py_files()
    test_add_lesson()
    test_add_lesson_invalid_category()
    test_get_status()
    test_unknown_tool()
    print("All tests passed ✓")
