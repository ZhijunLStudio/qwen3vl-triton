# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AICAS 2026 competition project for optimizing **Qwen3-VL-2B-Instruct** (Vision-Language Model) inference performance. The goal is to minimize TTFT (Time-To-First-Token) and maximize throughput while maintaining accuracy ≥ 95% of baseline.

**The only file participants should modify is `evaluation_wrapper.py`** — everything else is benchmark infrastructure.

## Environment Setup

```bash
source /data/lizhijun/anaconda3/bin/activate torch
export CUDA_VISIBLE_DEVICES=7
```

## Key Commands

```bash
# Run benchmark (measures TTFT + throughput + accuracy)
python benchmark.py --model-path ./Qwen3-VL-2B-Instruct --dataset-path ./data --output result.json --num-samples 20

# Quick benchmark with fewer samples
python benchmark.py --num-samples 5

# Evaluate accuracy separately from a result.json
python compute_accuracy.py --result result.json

# Profile decode performance
python profile_decode.py

# Run the autonomous optimization agent
cd optim && python main.py run
cd optim && python main.py run --max-iters 20
cd optim && python main.py status
cd optim && python main.py reset-baseline
cd optim && python main.py test
```

## Scoring Formula

```
score = accuracy × 0.40
      + (baseline_ttft - current_ttft) / baseline_ttft × 0.30
      + (current_throughput - baseline_throughput) / baseline_throughput × 0.30
```

Goals: TTFT < 30ms, Throughput > 400 tokens/s, Accuracy ≥ 95% of baseline.

## Architecture

### Main Competition Files

- **`evaluation_wrapper.py`** — The participant's core file. Contains `VLMModel` class that wraps Qwen3-VL-2B-Instruct. Loaded by `benchmark.py` and must expose `.processor`, `.model`, `.device` properties and a `.generate(image, question, max_new_tokens)` method.
- **`benchmark.py`** — Official benchmark harness. Do not modify. Measures TTFT (1-token generation) and throughput (128-token generation), then collects full answers for accuracy scoring.
- **`compute_accuracy.py`** — Evaluates `result.json` against dataset ground truth using class-label hit rate.

### Current Optimizations in `evaluation_wrapper.py`

1. **Triton W4A16 INT4 Quantization** (`SlimTritonINT4Linear`) — Custom Split-K GEMV kernel for decode, dequantize+matmul for prefill. Loads weights from `qwen3_2b_int4_fused_packed.pt`.
2. **Triton RMSNorm** — Custom Triton kernel replacing HuggingFace's default.
3. **TorchScript Fused RoPE** — `fused_rope_core` eliminates CPU dispatch overhead.
4. **Triton SwiGLU** — Fused gate+up activation for MLP layers.
5. **LLM Prefix KV Caching** — Caches KV for the visual token prefix (up to 3 images via LRU). On cache hit, skips visual re-encoding and injects the cached KV directly.
6. **Fast Decode Attention** — Custom decode-path attention bypassing FlashAttention2 for single-token generation; uses manual matmul with GQA support.

### OptimAgent (`optim/`)

An autonomous LLM-driven optimization loop that modifies `evaluation_wrapper.py` iteratively:

- **`optim/main.py`** — CLI entry point (`run`, `status`, `reset-baseline`, `test`)
- **`optim/config.py`** — All configuration: LLM endpoint (`InCoder-32B` at `192.168.97.11:8010`), goals, paths, benchmark commands
- **`optim/agent/loop.py`** — Agent loop: manages conversation history, calls tools, handles anomaly detection and commit decisions
- **`optim/agent/tools.py`** — Tool schemas (OpenAI function-calling format): `read`, `write`, `edit`, `bash`, `grep`, `glob`, `run_benchmark`, `commit_if_improved`, `revert_changes`, `add_lesson`, `get_status`
- **`optim/core/executor.py`** — Tool implementations (file I/O, benchmark execution, git operations)
- **`optim/core/metrics.py`** — `ScoreEngine`: score calculation and formatting
- **`optim/core/state.py`** — Persists baseline/current/best metrics to `.optim_state.json`
- **`optim/memory/`** — Three-tier memory: hot (last 5 full iterations), warm (LLM-compressed summary), cold (lessons learned in `.optim_memory.json`)
- **`optim/ui/tui.py`** — Rich terminal UI for live monitoring

### State Files (project root)

- `.optim_state.json` — Current optimization state (baseline/current/best scores + git hash)
- `.optim_memory.json` — Long-term lessons from the optimization agent
- `result.json` — Latest benchmark output

### Data & Model

- `./data/` — Validation dataset (HuggingFace `datasets` format, loaded with `load_from_disk`)
- `./Qwen3-VL-2B-Instruct/` — Model weights
- `qwen3_2b_int4_fused_packed.pt` — Pre-packed INT4 quantized weights (required for quantization path in `evaluation_wrapper.py`)
