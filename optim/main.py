"""main.py - Entry point for AICAS OptimAgent"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from config import GOALS
from core.metrics import ScoreEngine
from core.state import State
from memory.manager import Memory
from ui.tui import TUI
from agent.loop import AgentLoop


def cmd_run(args):
    state  = State()
    memory = Memory()
    tui    = TUI()

    log_cb = tui.make_log_cb(state)
    agent  = AgentLoop(state, memory, log_cb=log_cb, max_iters=args.max_iters)

    tui.start(state)
    try:
        agent.run()
    finally:
        tui.stop()

    # Final summary
    print("\n" + "=" * 60)
    cmd_status(args)


def cmd_status(args):
    state  = State()
    memory = Memory()
    s = state.data

    print("\n=== OptimAgent Status ===")
    print(f"Iteration  : {s.get('iteration', 0)}")
    print(f"Improvements: {s.get('improvements', 0)}")
    print(f"Consec fails: {s.get('consec_fails', 0)}")
    print(f"\nBaseline : {ScoreEngine.fmt(s.get('baseline', {}))}")
    print(f"Current  : {ScoreEngine.fmt(s.get('current',  {}))}")
    print(f"Best     : {ScoreEngine.fmt(s.get('best',     {}))}")
    print(f"Best commit: {s.get('best_git_hash', 'N/A')}")
    print(f"Goals met: {s.get('goals_achieved', False)}")
    print(f"\nLong-term memory:\n{memory.to_prompt_str(2000)}")
    print("=" * 60)


def cmd_reset(args):
    state = State()
    state.reset_baseline()
    print("Baseline cleared. Will re-measure on next run.")


def cmd_test(args):
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "tests.run_all"],
        cwd=os.path.dirname(__file__),
    )
    sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="AICAS OptimAgent — autonomous VLM optimization loop"
    )
    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Start optimization loop")
    p_run.add_argument("--max-iters", type=int, default=100, help="Max iterations (default 100)")
    p_run.add_argument("--no-tui",    action="store_true",   help="Plain text output")
    p_run.set_defaults(func=cmd_run)

    # status
    p_status = sub.add_parser("status", help="Show current optimization state")
    p_status.set_defaults(func=cmd_status)

    # reset
    p_reset = sub.add_parser("reset-baseline", help="Clear baseline and re-measure")
    p_reset.set_defaults(func=cmd_reset)

    # test
    p_test = sub.add_parser("test", help="Run test suite")
    p_test.set_defaults(func=cmd_test)

    args = parser.parse_args()

    if args.command is None:
        # Default: run with defaults
        args.max_iters = 100
        args.no_tui    = False
        cmd_run(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
