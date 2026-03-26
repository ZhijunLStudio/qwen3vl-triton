"""
config.py - All configuration for AICAS OptimAgent
Edit this file to change model endpoint, goals, paths, commands, etc.
"""

from pathlib import Path

# ─── LLM / API ─────────────────────────────────────────────────────────────

MODEL = {
    "base_url":   "http://192.168.97.11:8010/v1",
    "api_key":    "EMPTY",
    "name":       "InCoder-32B",
    "max_tokens": 8192,       # max tokens per response
    "temperature": 0.7,
    "timeout":    600,        # request timeout seconds
}

# Cheap/fast model for memory compression (can be same as MODEL)
COMPRESS_MODEL = {
    "base_url":   "http://192.168.97.11:8010/v1",
    "api_key":    "EMPTY",
    "name":       "InCoder-32B",
    "max_tokens": 1500,       # enough for a meaningful compression summary
    "temperature": 0.3,
    "timeout":    120,
}

# ─── Context Window ─────────────────────────────────────────────────────────

CONTEXT = {
    "window":              131072,  # 128k token context window (from opencode.json)
    "history_budget":      0.40,    # conversation history max 40% of window
    "compress_threshold":  0.30,    # compress when history tokens > 30% of window
    "max_hot_iters":       5,       # keep last N full iteration records in hot memory
    "max_compressed_chars": 8000,   # max chars for compressed warm memory
    "max_tool_output":     8000,    # max chars returned per tool call
    "max_file_read":       40000,   # max chars for file read
    "max_lessons":         32,      # max total lessons across all categories
}

# ─── Competition Goals ──────────────────────────────────────────────────────

GOALS = {
    "ttft_ms":              30.0,   # Time To First Token < 30ms
    "throughput":           400.0,  # Throughput > 400 tokens/s
    "acc_ratio":            0.95,   # Accuracy must stay >= 95% of baseline
    "min_commit_delta_ratio": 0.05, # Skip commit if improvement < 5% of current score (noise filter)
}

# ─── Anomaly / System-Load Detection ────────────────────────────────────────
# If benchmark results are abnormally worse than baseline (other processes on GPU/CPU),
# automatically wait and retry until results normalize.

ANOMALY = {
    "ttft_ratio":     1.5,   # flag if TTFT > baseline_ttft * 1.5 (50% slower)
    "tp_ratio":       0.6,   # flag if TP   < baseline_tp   * 0.6 (40% drop)
    "retry_interval": 300,   # seconds between retries (5 min)
    "max_retries":    6,     # give up after max_retries * retry_interval = 30 min
}

# Score formula weights (must sum to 1.0)
SCORE_WEIGHTS = {
    "accuracy":   0.40,
    "ttft":       0.30,
    "throughput": 0.30,
}

# ─── Paths ──────────────────────────────────────────────────────────────────

WORK_DIR      = Path("/data/lizhijun/work/AICAS/AICASGC")
WRAPPER_FILE  = WORK_DIR / "evaluation_wrapper.py"  # only file agent may modify
STATE_FILE    = WORK_DIR / ".optim_state.json"
MEMORY_FILE   = WORK_DIR / ".optim_memory.json"
LOG_DIR       = WORK_DIR / "optim" / "logs"
LOG_FILE      = LOG_DIR / "optim_log.jsonl"

# ─── Git / Network ───────────────────────────────────────────────────────────

# Proxy for git push (set to "" to disable)
HTTP_PROXY = "http://127.0.0.1:7870"

# ─── Commands ───────────────────────────────────────────────────────────────

# Activate conda env prefix (prepended to all bash commands)
CONDA_PREFIX = "source /data/lizhijun/anaconda3/bin/activate torch && "

BENCH_CMD = (
    f"{CONDA_PREFIX}"
    f"cd {WORK_DIR} && "
    "export CUDA_VISIBLE_DEVICES=7 && "
    "python benchmark.py "
    "--model-path ./Qwen3-VL-2B-Instruct "
    "--dataset-path ./data "
    "--output result.json "
    "--num-samples 20"
)

ACC_CMD = (
    f"{CONDA_PREFIX}"
    f"cd {WORK_DIR} && "
    "python compute_accuracy.py --result result.json"
)

# Default timeout for bash commands (seconds)
BASH_TIMEOUT = 60
# Override timeout for long-running commands
BENCH_TIMEOUT = 600
ACC_TIMEOUT   = 120
