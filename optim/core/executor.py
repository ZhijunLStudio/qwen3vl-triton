"""
core/executor.py - Tool implementation engine

Every tool in TOOLS (agent/tools.py) maps to a method here.
Score computation always done by ScoreEngine, never by LLM.
"""

import json
import re
import subprocess
import sys
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Callable, Optional, Dict, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    WORK_DIR, WRAPPER_FILE, BENCH_CMD, ACC_CMD,
    BASH_TIMEOUT, BENCH_TIMEOUT, ACC_TIMEOUT,
    CONTEXT, CONDA_PREFIX, GOALS, HTTP_PROXY, ANOMALY,
)
from core.metrics import ScoreEngine
from core.state import State


# ─── Shell helper ────────────────────────────────────────────────────────────

def _run(cmd: str, timeout: int = BASH_TIMEOUT) -> Tuple[bool, str]:
    """Run bash command, return (success, combined_output)."""
    try:
        r = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            executable="/bin/bash",
        )
        out = (r.stdout + r.stderr).strip()
        return r.returncode == 0, out
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout}s"
    except Exception as e:
        return False, f"subprocess error: {e}"


def _truncate(text: str, max_chars: int = CONTEXT["max_tool_output"]) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n\n...[truncated, {len(text)} chars total]...\n\n" + text[-half // 2:]


# ─── ToolExecutor ────────────────────────────────────────────────────────────

class ToolExecutor:
    """
    Implements all tools. Each public _tool_<name> method corresponds
    to a tool in agent/tools.py.

    log_cb: optional callable(message, level) for UI updates.
    """

    def __init__(
        self,
        state: State,
        memory,                   # memory.manager.Memory (avoid circular import)
        log_cb: Optional[Callable] = None,
    ):
        self.state  = state
        self.memory = memory
        self._log   = log_cb or (lambda msg, lv="info": print(f"[{lv}] {msg}"))

    def execute(self, name: str, inputs: Dict[str, Any]) -> str:
        """Dispatch tool call by name. Returns string result."""
        method = getattr(self, f"_tool_{name}", None)
        if method is None:
            return f"ERROR: Unknown tool '{name}'"
        try:
            return method(**inputs)
        except TypeError as e:
            return f"ERROR: Bad arguments for '{name}': {e}"
        except Exception as e:
            return f"ERROR: Tool '{name}' raised {type(e).__name__}: {e}"

    # ── read ─────────────────────────────────────────────────────────────────

    def _tool_read(self, path: str, offset: int = 0, limit: int = 0) -> str:
        """Read file contents, with optional line offset/limit."""
        full = WORK_DIR / path
        if not full.exists():
            return f"ERROR: File not found: {path}"
        try:
            text = full.read_text(encoding="utf-8")
            lines = text.splitlines()
            if offset:
                lines = lines[offset:]
            if limit:
                lines = lines[:limit]
            content = "\n".join(lines)
            return _truncate(content, CONTEXT["max_file_read"])
        except Exception as e:
            return f"ERROR reading {path}: {e}"

    # ── write ────────────────────────────────────────────────────────────────

    def _tool_write(self, path: str, content: str) -> str:
        """Write (create or overwrite) a file."""
        if path != "evaluation_wrapper.py":
            return f"ERROR: Only evaluation_wrapper.py may be written. Got: {path}"
        full = WORK_DIR / path
        try:
            full.write_text(content, encoding="utf-8")
            lines = content.count("\n") + 1
            self._log(f"write {path} ({lines} lines)", "tool")
            return f"OK: wrote {path} ({lines} lines, {len(content)} chars)"
        except Exception as e:
            return f"ERROR writing {path}: {e}"

    # ── edit ─────────────────────────────────────────────────────────────────

    def _tool_edit(self, path: str, old_str: str, new_str: str) -> str:
        """
        Targeted in-place edit: replace first occurrence of old_str with new_str.
        old_str must be unique in the file. Safer than full rewrite.
        """
        if path != "evaluation_wrapper.py":
            return f"ERROR: Only evaluation_wrapper.py may be edited. Got: {path}"
        full = WORK_DIR / path
        if not full.exists():
            return f"ERROR: File not found: {path}"
        try:
            original = full.read_text(encoding="utf-8")
            count = original.count(old_str)
            if count == 0:
                # Show closest match hint
                return (
                    f"ERROR: old_str not found in {path}.\n"
                    f"old_str was ({len(old_str)} chars):\n{old_str[:200]}"
                )
            if count > 1:
                return (
                    f"ERROR: old_str appears {count} times in {path} — must be unique.\n"
                    f"Provide more context around the target."
                )
            updated = original.replace(old_str, new_str, 1)
            full.write_text(updated, encoding="utf-8")
            self._log(f"edit {path} ({len(old_str)} → {len(new_str)} chars)", "tool")
            return f"OK: applied edit to {path}"
        except Exception as e:
            return f"ERROR editing {path}: {e}"

    # ── bash ─────────────────────────────────────────────────────────────────

    def _tool_bash(self, command: str, timeout: int = BASH_TIMEOUT) -> str:
        """Execute a shell command inside conda torch env."""
        full_cmd = f"{CONDA_PREFIX}cd {WORK_DIR} && {command}"
        self._log(f"bash: {command[:80]}", "tool")
        ok, out = _run(full_cmd, timeout=min(timeout, 300))
        result = out if out else ("(exit 0, no output)" if ok else "(exit non-zero, no output)")
        return _truncate(result)

    # ── grep ─────────────────────────────────────────────────────────────────

    def _tool_grep(
        self,
        pattern: str,
        path: str = ".",
        glob: str = "*.py",
        case_sensitive: bool = True,
    ) -> str:
        """Search for a regex pattern in files."""
        flags = "" if case_sensitive else " -i"
        full_path = WORK_DIR / path
        cmd = (
            f"grep -rn{flags} --include='{glob}' "
            f"-E '{pattern}' '{full_path}' 2>&1 | head -60"
        )
        ok, out = _run(cmd, timeout=15)
        return _truncate(out or "(no matches)") if out else "(no matches)"

    # ── glob ─────────────────────────────────────────────────────────────────

    def _tool_glob(self, pattern: str, path: str = ".") -> str:
        """List files matching a glob pattern."""
        import glob as _glob
        base = str(WORK_DIR / path)
        full_pattern = str(WORK_DIR / path / pattern)
        matches = _glob.glob(full_pattern, recursive=True)
        # Make paths relative to WORK_DIR
        rel = []
        for m in sorted(matches)[:80]:
            try:
                rel.append(str(Path(m).relative_to(WORK_DIR)))
            except ValueError:
                rel.append(m)
        return "\n".join(rel) if rel else "(no matches)"

    # ── webfetch ─────────────────────────────────────────────────────────────

    def _tool_webfetch(self, url: str) -> str:
        """Fetch a URL and return text content (HTML stripped)."""
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 OptimAgent/1.0"}
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read().decode("utf-8", errors="replace")

            # Strip HTML tags
            text = re.sub(r"<[^>]+>", " ", raw)
            text = re.sub(r"\s+", " ", text).strip()
            self._log(f"webfetch {url[:60]}", "tool")
            return _truncate(text, 4000)
        except Exception as e:
            return f"ERROR fetching {url}: {e}"

    # ── run_benchmark ─────────────────────────────────────────────────────────
    # ★ Score is computed HERE by ScoreEngine, NOT by the LLM ★

    def _run_single_benchmark(self) -> Tuple[bool, Optional[Dict]]:
        """
        Execute one benchmark + accuracy run.
        Returns (ok, metrics_dict) where metrics_dict has ttft_ms/throughput/accuracy,
        or (False, None) on hard failure.
        """
        ok, bench_out = _run(BENCH_CMD, timeout=BENCH_TIMEOUT)
        if not ok:
            self._log("Benchmark process FAILED", "error")
            self._last_bench_error = bench_out
            return False, None

        result_path = WORK_DIR / "result.json"
        try:
            result = json.loads(result_path.read_text())
            perf   = result.get("performance", {})
            ttft   = perf.get("avg_ttft_ms")
            tp     = perf.get("avg_throughput_tokens_per_sec", 0.0)
        except Exception as e:
            self._last_bench_error = f"Cannot parse result.json: {e}"
            return False, None

        if ttft is None:
            self._last_bench_error = f"TTFT is None — all samples failed\n{bench_out[-500:]}"
            return False, None

        self._log("Computing accuracy...", "bench")
        _, acc_out = _run(ACC_CMD, timeout=ACC_TIMEOUT)
        acc = self._parse_accuracy(acc_out)

        return True, {
            "ttft_ms":    float(ttft),
            "throughput": float(tp),
            "accuracy":   acc,
        }

    def _is_anomalous(self, metrics: Dict) -> Tuple[bool, str]:
        """
        Detect if metrics suggest the machine is under load (other processes
        competing for GPU/CPU). Compares against stored baseline.
        Returns (anomalous, reason_string).
        """
        baseline = self.state.get("baseline")
        if not baseline:
            return False, ""   # no reference yet — accept as-is

        b_ttft = baseline.get("ttft_ms", 0.0)
        b_tp   = baseline.get("throughput", 0.0)
        c_ttft = metrics.get("ttft_ms", 0.0)
        c_tp   = metrics.get("throughput", 0.0)

        ttft_thresh = ANOMALY["ttft_ratio"]
        tp_thresh   = ANOMALY["tp_ratio"]

        if b_ttft > 0 and c_ttft > b_ttft * ttft_thresh:
            ratio = c_ttft / b_ttft
            return True, (
                f"TTFT {c_ttft:.1f}ms is {ratio:.1f}x the baseline "
                f"{b_ttft:.1f}ms (threshold >{ttft_thresh}x)"
            )
        if b_tp > 0 and c_tp < b_tp * tp_thresh:
            ratio = c_tp / b_tp
            return True, (
                f"Throughput {c_tp:.1f}t/s is {ratio:.2f}x the baseline "
                f"{b_tp:.1f}t/s (threshold <{tp_thresh}x)"
            )
        return False, ""

    def _tool_run_benchmark(self) -> str:
        """
        Run benchmark + accuracy test with automatic system-load retry.
        If results are anomalous (machine under heavy load), waits
        ANOMALY.retry_interval seconds and retries up to ANOMALY.max_retries times.
        Score is computed by ScoreEngine (Python), never by LLM.
        """
        self._last_bench_error = ""
        max_retries    = ANOMALY["max_retries"]
        retry_interval = ANOMALY["retry_interval"]

        raw_metrics: Optional[Dict] = None
        for attempt in range(max_retries + 1):
            self._log(
                f"Running benchmark (2-5 min)..."
                + (f" [attempt {attempt+1}/{max_retries+1}]" if attempt > 0 else ""),
                "bench"
            )
            ok, raw_metrics = self._run_single_benchmark()

            if not ok:
                snippet = (self._last_bench_error or "")[-1500:]
                return json.dumps({
                    "status": "FAILED",
                    "reason": "Benchmark process exited with error",
                    "output": snippet,
                    "action": "Fix the code error, then call run_benchmark again",
                }, ensure_ascii=False)

            anomalous, reason = self._is_anomalous(raw_metrics)
            if not anomalous:
                break

            if attempt < max_retries:
                self._log(
                    f"⚠ System load detected: {reason}. "
                    f"Waiting {retry_interval}s before retry...",
                    "warn"
                )
                time.sleep(retry_interval)
            else:
                self._log(
                    f"⚠ System load NOT resolved after {max_retries} retries "
                    f"({max_retries * retry_interval // 60} min). "
                    f"Proceeding with caution.",
                    "warn"
                )
                # Tag the metrics as potentially unreliable
                raw_metrics["_system_load_warning"] = reason

        # ─── SCORE COMPUTED BY PYTHON, NOT LLM ────────────────────────────
        new_metrics = {
            "ttft_ms":    raw_metrics["ttft_ms"],
            "throughput": raw_metrics["throughput"],
            "accuracy":   raw_metrics["accuracy"],
        }
        baseline = self.state.get("baseline", {})
        current  = self.state.get("current",  {})

        new_metrics["score"] = ScoreEngine.compute(new_metrics, baseline)
        delta = new_metrics["score"] - current.get("score", 0.0)

        valid, valid_reason = ScoreEngine.is_accuracy_valid(new_metrics, baseline)
        improved = valid and delta > 0
        goals_met = ScoreEngine.goals_met(new_metrics)

        # Cache result for commit_if_improved
        self.state.set("last_bench", {
            "metrics":       new_metrics,
            "valid":         valid,
            "valid_reason":  valid_reason,
            "delta":         delta,
            "improved":      improved,
        })

        self._log(
            f"{ScoreEngine.fmt(new_metrics)}  "
            f"{ScoreEngine.fmt_delta(delta)}",
            "result"
        )

        system_load_warning = raw_metrics.get("_system_load_warning", "")

        return json.dumps({
            "status":               "SUCCESS",
            "metrics":              new_metrics,
            "baseline":             {k: baseline.get(k) for k in ["ttft_ms", "throughput", "accuracy", "score"]},
            "delta_score":          round(delta, 6),
            "accuracy_valid":       valid,
            "accuracy_reason":      valid_reason,
            "is_improvement":       improved,
            "goals_met":            goals_met,
            "system_load_warning":  system_load_warning,
            "goals_status": {
                "ttft":       f"{new_metrics['ttft_ms']:.1f}ms (goal: <{30}ms)",
                "throughput": f"{new_metrics['throughput']:.1f} t/s (goal: >{400})",
            },
            "next_action": (
                "Call commit_if_improved to save progress"
                if improved else
                "Call revert_changes to undo, then add_lesson with what failed"
            ),
        }, indent=2, ensure_ascii=False)

    def _parse_accuracy(self, output: str) -> float:
        """Parse accuracy from compute_accuracy.py output."""
        # Primary: "类别命中率: XX.XX%"  or  "hit_rate: XX.XX%"
        for pattern in [
            r'hit.?rate[^\d]*([\d.]+)%',
            r'\u7c7b\u522b\u547d\u4e2d\u7387[^\d]*([\d.]+)%',
            r'accuracy[^\d]*([\d.]+)%',
        ]:
            m = re.search(pattern, output, re.IGNORECASE)
            if m:
                return float(m.group(1)) / 100
        # Fallback: hits/total from "命中数: N" and "总样本数: N"
        hits_m  = re.search(r'\u547d\u4e2d\u6570[^\d]*(\d+)', output)
        total_m = re.search(r'\u603b\u6837\u672c\u6570[^\d]*(\d+)', output)
        if hits_m and total_m:
            hits  = int(hits_m.group(1))
            total = int(total_m.group(1))
            return hits / total if total > 0 else 0.0
        return 0.0

    # ── commit_if_improved ───────────────────────────────────────────────────

    def _tool_commit_if_improved(self) -> str:
        """
        Commit and push if last benchmark showed improvement.
        Reverts automatically if not improved.
        """
        bench = self.state.get("last_bench")
        if not bench:
            return "ERROR: No benchmark result available. Call run_benchmark first."

        metrics  = bench["metrics"]
        valid    = bench["valid"]
        delta    = bench["delta"]
        improved = bench["improved"]

        if not valid:
            self._tool_revert_changes()
            return (
                f"REVERTED: Accuracy constraint violated — {bench.get('valid_reason')}\n"
                "Recommendation: add_lesson with what caused accuracy drop."
            )

        if delta <= 0:
            self._tool_revert_changes()
            return (
                f"REVERTED: No score improvement (delta={delta:+.4f})\n"
                "Recommendation: add_lesson with why this approach failed."
            )

        # ── Noise filter: ignore tiny improvements (likely TTFT/TP fluctuation) ──
        current_score = self.state.get("current", {}).get("score", 0.0)
        baseline_score = self.state.get("baseline", {}).get("score", 0.0)
        ref_score = max(current_score, baseline_score, 1e-9)
        min_delta = ref_score * GOALS.get("min_commit_delta_ratio", 0.05)
        if delta < min_delta:
            self._tool_revert_changes()
            return (
                f"SKIPPED: Improvement too small (delta={delta:+.4f}, min={min_delta:.4f}). "
                f"Likely TTFT/throughput measurement noise. Reverted.\n"
                "Recommendation: try a more impactful optimization."
            )

        # ── Commit ──────────────────────────────────────────────────────────
        sc   = metrics["score"]
        ttft = metrics["ttft_ms"]
        tp   = metrics["throughput"]
        acc  = metrics["accuracy"]
        msg  = (
            f"optim: TTFT={ttft:.1f}ms TP={tp:.1f}t/s "
            f"Acc={acc:.1%} Score={sc:.4f} delta={delta:+.4f}"
        )

        _run(f'cd {WORK_DIR} && git add evaluation_wrapper.py', 30)
        ok, out = _run(f'cd {WORK_DIR} && git commit -m "{msg}"', 30)

        if not ok:
            if "nothing to commit" in out:
                return "SKIPPED: Nothing changed since last commit."
            return f"ERROR: git commit failed:\n{out}"

        ok2, rev = _run(f'cd {WORK_DIR} && git rev-parse HEAD', 10)
        git_hash = rev.strip()[:8] if ok2 else "unknown"

        proxy_prefix = (
            f"http_proxy={HTTP_PROXY} https_proxy={HTTP_PROXY} "
            if HTTP_PROXY else ""
        )
        ok3, _ = _run(f'cd {WORK_DIR} && {proxy_prefix}git push origin main', 60)
        push_status = "pushed" if ok3 else "push failed (check remote)"

        # ── Create a git tag for milestone tracking ──────────────────────────
        improvements_n = self.state.get("improvements", 0) + 1
        tag_name = f"optim-v{improvements_n}-score{sc:.4f}"
        _run(f'cd {WORK_DIR} && git tag {tag_name}', 10)
        _run(f'cd {WORK_DIR} && {proxy_prefix}git push origin {tag_name}', 30)

        # Update state
        self.state.update({
            "current":       metrics,
            "improvements":  self.state.get("improvements", 0) + 1,
            "consec_fails":  0,
            "best_git_hash": git_hash,
        })
        if sc > self.state.get("best", {}).get("score", 0):
            self.state.set("best", metrics)

        if ScoreEngine.goals_met(metrics):
            self.state.set("goals_achieved", True)

        self._log(f"Committed {git_hash} ({push_status})", "git")

        # Auto-lesson for successful commit
        self.memory.add(
            "worked",
            f"commit {git_hash} delta={delta:+.4f}: "
            f"TTFT={ttft:.1f}ms TP={tp:.1f}t/s"
        )

        return (
            f"COMMITTED: {git_hash} ({push_status})\n"
            f"Score delta: {delta:+.4f}\n"
            f"New metrics: {ScoreEngine.fmt(metrics)}"
        )

    # ── revert_changes ───────────────────────────────────────────────────────

    def _tool_revert_changes(self) -> str:
        """Revert evaluation_wrapper.py to last git-committed version."""
        ok, out = _run(
            f'cd {WORK_DIR} && git checkout evaluation_wrapper.py', 30
        )
        self.state.increment("consec_fails")
        self._log("Reverted evaluation_wrapper.py", "tool")
        return "OK: reverted evaluation_wrapper.py to last commit" if ok else f"ERROR: {out}"

    # ── add_lesson ───────────────────────────────────────────────────────────

    def _tool_add_lesson(self, category: str, content: str) -> str:
        """Add a lesson to persistent long-term memory."""
        valid = {"worked", "failed", "observation", "strategy"}
        if category not in valid:
            return f"ERROR: category must be one of {valid}, got '{category}'"
        self.memory.add(category, content)
        self._log(f"lesson [{category}]: {content[:60]}", "memory")
        return f"OK: stored lesson in [{category}]"

    # ── get_status ───────────────────────────────────────────────────────────

    def _tool_get_status(self) -> str:
        """Return current optimization state as JSON."""
        s = self.state.data
        hot = s.get("hot_history", [])
        recent = [
            {
                "iter":    r.get("iter"),
                "outcome": r.get("outcome"),
                "delta":   r.get("delta"),
                "action":  r.get("action", "")[:100],
            }
            for r in hot[-5:]
        ]
        return json.dumps({
            "iteration":      s.get("iteration", 0),
            "total_attempts": s.get("total_attempts", 0),
            "improvements":   s.get("improvements", 0),
            "consec_fails":   s.get("consec_fails", 0),
            "goals_achieved": s.get("goals_achieved", False),
            "best_git_hash":  s.get("best_git_hash", ""),
            "baseline":       s.get("baseline", {}),
            "current":        s.get("current", {}),
            "best":           s.get("best", {}),
            "goals":          {
                "ttft_target":       30.0,
                "throughput_target": 400.0,
                "acc_ratio_min":     0.95,
            },
            "recent_history":  recent,
            "warm_summary":    s.get("warm_summary", ""),
            "long_term_memory": self.memory.to_prompt_str(1500),
        }, indent=2, ensure_ascii=False)
