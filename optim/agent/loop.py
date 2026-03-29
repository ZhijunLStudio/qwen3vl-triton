"""
agent/loop.py - Main agent loop using OpenAI-compatible tool calling

Each iteration:
  1. Build system prompt from state + memory
  2. Multi-turn tool-use conversation with LLM
  3. Tools execute in Python (score always computed by ScoreEngine)
  4. Update hot history, check goals, maybe compress context
"""

import json
import sys
import os
import time
from datetime import datetime
from typing import Optional, Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import MODEL, CONTEXT, GOALS, WORK_DIR, LOG_DIR, LOG_FILE

LOG_DIR.mkdir(parents=True, exist_ok=True)  # ensure logs folder exists
from core.metrics import ScoreEngine
from core.state import State
from core.executor import ToolExecutor
from memory.manager import Memory
from memory.compressor import ContextCompressor
from agent.tools import TOOLS

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)


# ─── System prompt (English) ─────────────────────────────────────────────────

def build_system_prompt(state: State, memory: Memory) -> str:
    s        = state.data
    baseline = s.get("baseline", {})
    best     = s.get("best",     {})
    current  = s.get("current",  {})
    lessons  = memory.to_prompt_str(2000)
    warm     = s.get("warm_summary", "")

    return f"""You are an expert ML inference optimization engineer specializing in VLM deployment.

## Mission
Optimize `evaluation_wrapper.py` for Qwen3-VL-2B-Instruct to maximize competition score.

## Competition Goals
- TTFT (Time To First Token): < {GOALS['ttft_ms']}ms  (current baseline: {baseline.get('ttft_ms', 'N/A')}ms)
- Throughput: > {GOALS['throughput']} tokens/s  (current baseline: {baseline.get('throughput', 'N/A')} t/s)
- Accuracy: must stay >= {GOALS['acc_ratio']:.0%} of baseline

## Scoring Formula  (computed automatically, NOT by you)
score = accuracy_ratio × 0.40 + ttft_improvement × 0.30 + throughput_improvement × 0.30

## Current State
- Baseline: {ScoreEngine.fmt(baseline) if baseline else 'not measured yet'}
- Current : {ScoreEngine.fmt(current)  if current  else 'not measured yet'}
- Best    : {ScoreEngine.fmt(best)     if best     else 'not measured yet'}
- Iteration: {s.get('iteration', 0)}  |  Improvements: {s.get('improvements', 0)}  |  Consec. fails: {s.get('consec_fails', 0)}

## Strict Rules
1. Only `evaluation_wrapper.py` may be modified (read-only access to all other files)
2. `VLMModel` class must remain and expose `.processor`, `.model`, `.device` attributes
3. Make ONE focused change per iteration — easier to debug
4. After every code change: call `run_benchmark` to validate
5. After benchmark: call `commit_if_improved` OR `revert_changes`
6. After every attempt (win or fail): call `add_lesson`
7. **NEVER compute the score yourself** — it is always provided in the run_benchmark result

## ❌ PROVEN FAILURES — DO NOT TRY THESE AGAIN
These have been tested and all degrade performance on this hardware:
- torch.backends.cuda.matmul.allow_tf32 / set_float32_matmul_precision / cudnn.benchmark
- Flash Attention 2 (attn_implementation='flash_attention_2') — incompatible with this model
- torch.compile with any mode (reduce-overhead, max-autotune) — always degrades
- BitsAndBytes INT8 / INT4 quantization — breaks generation (TTFT=None)

## ✅ Untried Strategies (try these in order)

**A. CUDA Graph capture for decode (highest impact)**
   - After loading model, do a warmup run then capture CUDA graph:
     ```python
     # In __init__, after model load:
     self._warmup()
     ```
   - Use `torch.cuda.CUDAGraph()` to capture the forward pass
   - This eliminates Python overhead on each token generation

**B. Static KV Cache (reduces memory allocation overhead)**
   - Use `transformers.StaticCache` instead of DynamicCache
   - Pre-allocate KV cache at init time:
     ```python
     from transformers import StaticCache
     self._model.generation_config.cache_implementation = "static"
     ```
   - Combine with `torch.compile(model.forward, mode='reduce-overhead')`

**C. Warmup run at init time**
   - Add a dummy generation in `__init__` to pre-compile CUDA kernels
   - Dramatically reduces TTFT for the first real request
   - Use a short dummy input (1 token) to warm up

**D. torch.inference_mode() instead of torch.no_grad()**
   - `torch.inference_mode()` is strictly faster (disables autograd version counter)
   - Replace: `with torch.no_grad():` → `with torch.inference_mode():`

**E. Optimize image preprocessing**
   - Cache processor output if same image appears multiple times
   - Move image preprocessing to GPU if possible
   - Use `torch.float16` for image tensors explicitly

**F. Reduce Python overhead in generate()**
   - Remove unnecessary dict lookups
   - Pre-compute `pad_token_id` at init time, not every call
   - Use `inputs_embeds` path to skip re-tokenization overhead

**G. Batch inference with dynamic batching**
   - If benchmark uses sequential calls, try batching requests
   - Throughput improves dramatically with batch_size > 1

## Workflow Per Iteration
1. `get_status` → understand current state and history
2. `read` key files to identify bottleneck
3. `bash` to profile if needed (e.g., python profile_decode.py)
4. `edit` or `write` evaluation_wrapper.py with ONE change
5. `run_benchmark` → system computes score, shows delta
6. `commit_if_improved` or `revert_changes`
7. `add_lesson` with what you learned

## Long-Term Memory (from previous iterations)
{lessons if lessons else '(no lessons yet — start fresh)'}

## Compressed History (older iterations)
{warm if warm else '(no prior history)'}
"""


# ─── Token budget helpers ────────────────────────────────────────────────────

def token_est(text: str) -> int:
    zh = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return int(zh / 1.5 + (len(text) - zh) / 4)


def messages_tokens(messages: list) -> int:
    return sum(token_est(json.dumps(m)) for m in messages)


def trim_messages(messages: list, budget: int) -> list:
    """
    Trim oldest message pairs when conversation history exceeds budget.
    Always keeps the first user message (context seed).
    """
    while messages_tokens(messages) > budget and len(messages) > 2:
        # Remove the oldest user+assistant pair (index 1 and 2)
        if len(messages) > 2:
            messages.pop(1)
            if len(messages) > 1 and messages[1].get("role") == "assistant":
                messages.pop(1)
    return messages


# ─── Agent Loop ──────────────────────────────────────────────────────────────

class AgentLoop:
    """
    Runs the iterative optimization loop using OpenAI tool calling API.

    Each call to run_iteration():
      - Starts a fresh conversation seeded with context
      - Allows LLM to call tools repeatedly
      - Ends when LLM stops calling tools (finish_reason='stop')
      - Records the iteration outcome
    """

    MAX_TOOL_TURNS = 25   # safety limit per iteration

    def __init__(
        self,
        state:    State,
        memory:   Memory,
        log_cb:   Optional[Callable] = None,
        max_iters: int = 100,
    ):
        self.state      = state
        self.memory     = memory
        self.max_iters  = max_iters
        self._log       = log_cb or (lambda msg, lv="info": print(f"[{lv.upper()}] {msg}"))
        self.compressor = ContextCompressor(state)
        self.executor   = ToolExecutor(state, memory, log_cb)
        self.client     = OpenAI(
            base_url=MODEL["base_url"],
            api_key=MODEL["api_key"],
        )
        self._history_budget = int(CONTEXT["window"] * CONTEXT["history_budget"])

    # ── Baseline measurement ─────────────────────────────────────────────────

    def measure_baseline(self):
        self._log("Measuring baseline performance...", "info")
        result_str = self.executor._tool_run_benchmark()
        data = json.loads(result_str)

        if data.get("status") != "SUCCESS":
            self._log(f"Baseline failed: {data.get('reason')}", "error")
            sys.exit(1)

        baseline = data["metrics"]
        # Compute baseline score vs itself: only accuracy term is non-zero
        # (TTFT and TP improvements are both 0 at baseline)
        baseline["score"] = ScoreEngine.compute(baseline, baseline)
        current = dict(baseline)
        self.state.update({
            "baseline": baseline,
            "current":  current,
            "best":     dict(baseline),
        })
        self._log(f"Baseline: {ScoreEngine.fmt(baseline)}", "result")

    # ── Single iteration ─────────────────────────────────────────────────────

    def run_iteration(self) -> bool:
        """
        Run one full optimization iteration.
        Returns True to continue, False to stop.
        """
        self.state.increment("iteration")
        self.state.increment("total_attempts")
        iter_n = self.state.get("iteration")

        self._log(f"{'─'*50}", "info")
        self._log(f"Iteration #{iter_n}", "info")
        self._log(ScoreEngine.fmt(self.state.get("current", {})), "info")

        # Context management: compress if needed
        self.compressor.maybe_compress()

        # Build initial seed message for this iteration
        s    = self.state.data
        warm = s.get("warm_summary", "")
        hot  = s.get("hot_history", [])

        seed_parts = [
            "Start a new optimization iteration.",
            "",
            "Call get_status first to review the full state, "
            "then decide on your optimization approach.",
        ]
        if warm:
            seed_parts += ["", f"### Prior History Summary\n{warm}"]
        if hot:
            recent_str = json.dumps(hot[-3:], indent=2)
            seed_parts += ["", f"### Last 3 Iterations\n{recent_str}"]

        messages = [{"role": "user", "content": "\n".join(seed_parts)}]

        # ── Multi-turn tool-use loop ─────────────────────────────────────────
        action_summary = ""
        outcome        = "unknown"
        delta          = 0.0
        turn           = 0

        bench_called   = False  # has run_benchmark been called this iteration?
        bash_count     = 0     # bash exploration counter

        while turn < self.MAX_TOOL_TURNS:
            turn += 1

            # ── Urgency injection ────────────────────────────────────────────
            # If we're burning turns on exploration without making a code change,
            # inject a user message to force action.
            remaining = self.MAX_TOOL_TURNS - turn
            if remaining == 8 and not bench_called and not action_summary:
                messages.append({
                    "role": "user",
                    "content": (
                        f"⚠️ URGENT: {remaining} tool turns remaining. "
                        "You have NOT yet called run_benchmark this iteration. "
                        "Stop exploring — commit to ONE code change right now: "
                        "call edit or write, then run_benchmark, then commit_if_improved or revert_changes. "
                        "If you cannot decide, revert and call add_lesson."
                    ),
                })
            elif remaining == 3 and not bench_called:
                messages.append({
                    "role": "user",
                    "content": (
                        f"⚠️ CRITICAL: only {remaining} turns left and no benchmark run yet. "
                        "Call revert_changes immediately, then add_lesson explaining what you explored."
                    ),
                })

            # Trim conversation if approaching budget
            messages = trim_messages(messages, self._history_budget)

            try:
                response = self.client.chat.completions.create(
                    model=MODEL["name"],
                    messages=[
                        {"role": "system", "content": build_system_prompt(self.state, self.memory)},
                        *messages,
                    ],
                    tools=TOOLS,
                    tool_choice="auto",
                    max_tokens=MODEL["max_tokens"],
                    temperature=MODEL["temperature"],
                    timeout=MODEL["timeout"],
                )
            except Exception as e:
                self._log(f"LLM API error: {e}", "error")
                time.sleep(10)
                break

            assistant_msg = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # Log any text content
            if assistant_msg.content:
                snippet = (assistant_msg.content or "").strip()[:120].replace("\n", " ")
                if snippet:
                    self._log(f"💭 {snippet}", "think")

            # Append assistant message to conversation
            messages.append({
                "role":       "assistant",
                "content":    assistant_msg.content or "",
                "tool_calls": [
                    {
                        "id":   tc.id,
                        "type": "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in (assistant_msg.tool_calls or [])
                ] or None,
            })
            # Clean up None
            if messages[-1]["tool_calls"] is None:
                del messages[-1]["tool_calls"]

            if finish_reason == "tool_calls" and assistant_msg.tool_calls:
                tool_results = []

                for tc in assistant_msg.tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}

                    self._log(f"🔧 {name}({list(args.keys())})", "tool")

                    if name == "bash":
                        bash_count += 1

                    result_str = self.executor.execute(name, args)

                    # Truncate large results before feeding back
                    if len(result_str) > CONTEXT["max_tool_output"]:
                        half = CONTEXT["max_tool_output"] // 2
                        result_str = (
                            result_str[:half] +
                            f"\n...[truncated]...\n" +
                            result_str[-half // 2:]
                        )

                    tool_results.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      result_str,
                    })

                    # Track outcomes
                    if name in ("write", "edit"):
                        action_summary = f"{name}: {str(args)[:80]}"

                    if name == "run_benchmark":
                        bench_called = True
                        try:
                            d = json.loads(result_str)
                            if d.get("status") == "SUCCESS":
                                delta   = d.get("delta_score", 0.0)
                                outcome = "improved" if d.get("is_improvement") else "degraded"
                                self._log(
                                    f"{'✅' if d.get('is_improvement') else '❌'} "
                                    f"{ScoreEngine.fmt(d['metrics'])}  "
                                    f"{ScoreEngine.fmt_delta(delta)}",
                                    "result"
                                )
                        except Exception:
                            pass

                    if name == "commit_if_improved" and "COMMITTED" in result_str:
                        self._log(result_str.split("\n")[0], "git")

                messages.extend(tool_results)

            elif finish_reason == "stop":
                self._log("Iteration complete", "info")
                break

            else:
                self._log(f"Unexpected finish_reason: {finish_reason}", "error")
                break

        # ── Record iteration in hot history ──────────────────────────────────
        self.state.append_hot({
            "iter":    iter_n,
            "outcome": outcome,
            "delta":   round(delta, 6),
            "action":  action_summary,
            "ts":      datetime.now().strftime("%H:%M"),
        })

        # Write to log file
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "iter":    iter_n,
                "ts":      datetime.now().isoformat(),
                "outcome": outcome,
                "delta":   delta,
                "action":  action_summary,
            }, ensure_ascii=False) + "\n")

        # Check termination
        if self.state.get("goals_achieved"):
            self._log("🎉 All goals achieved!", "result")
            return False

        return True

    # ── Main run loop ────────────────────────────────────────────────────────

    def run(self):
        # Ensure baseline exists
        if not self.state.get("baseline"):
            self.measure_baseline()

        self._log("OptimAgent started", "info")
        self._log(
            f"Goals: TTFT < {GOALS['ttft_ms']}ms  TP > {GOALS['throughput']} t/s",
            "info"
        )
        self._log(f"Baseline: {ScoreEngine.fmt(self.state.get('baseline', {}))}", "info")
        self._log(f"Best so far: {ScoreEngine.fmt(self.state.get('best', {}))}", "info")

        for i in range(self.max_iters):
            try:
                cont = self.run_iteration()
            except KeyboardInterrupt:
                self._log("Interrupted by user", "info")
                break
            except Exception as e:
                self._log(f"Unexpected error: {e}", "error")
                import traceback
                traceback.print_exc()
                time.sleep(5)
                cont = True

            if not cont:
                break
            time.sleep(2)

        self._log("Optimization finished", "info")
        self._log(f"Best: {ScoreEngine.fmt(self.state.get('best', {}))}", "result")
