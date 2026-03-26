"""tests/run_all.py - Run all test modules and report results"""
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

TEST_MODULES = [
    ("test_metrics",  "core/metrics.py"),
    ("test_state",    "core/state.py"),
    ("test_memory",   "memory/manager.py"),
    ("test_executor", "core/executor.py  (no GPU required)"),
]


def run_module(module_name: str) -> tuple:
    """Import and run all test_* functions. Returns (passed, failed, errors)."""
    import importlib
    passed = failed = 0
    errors = []

    try:
        mod = importlib.import_module(f"tests.{module_name}")
    except Exception as e:
        return 0, 1, [f"Import failed: {e}\n{traceback.format_exc()}"]

    fns = [v for k, v in vars(mod).items() if k.startswith("test_") and callable(v)]

    for fn in fns:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            failed += 1
            errors.append(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            errors.append(f"ERROR {fn.__name__}: {e}\n{traceback.format_exc()}")

    return passed, failed, errors


def main():
    print("=" * 60)
    print("  AICAS OptimAgent - Test Suite")
    print("=" * 60)

    total_pass = total_fail = 0

    for module_name, description in TEST_MODULES:
        print(f"\n[{module_name}]  ({description})")
        passed, failed, errors = run_module(module_name)
        total_pass += passed
        total_fail += failed

        if errors:
            for e in errors:
                print(f"  ✗ {e}")
        if passed and not errors:
            print(f"  All {passed} tests passed")
        elif passed:
            print(f"  {passed} passed, {failed} failed")

    print("\n" + "=" * 60)
    status = "✓ ALL PASSED" if total_fail == 0 else f"✗ {total_fail} FAILED"
    print(f"  {status}  ({total_pass} passed, {total_fail} failed)")
    print("=" * 60)

    sys.exit(1 if total_fail > 0 else 0)


if __name__ == "__main__":
    main()
