"""
memory/compressor.py - LLM-based history compression (Warm tier)

Compresses N old iteration records into a compact summary paragraph,
freeing up context budget for fresh iterations.
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import COMPRESS_MODEL, CONTEXT
from core.state import State


def _make_client():
    try:
        from openai import OpenAI
        return OpenAI(
            base_url=COMPRESS_MODEL["base_url"],
            api_key=COMPRESS_MODEL["api_key"],
        )
    except ImportError:
        return None


def compress_iterations(records: list, client=None) -> str:
    """
    Compress a list of iteration records into a concise text summary.

    Tries LLM compression first; falls back to rule-based if unavailable.
    """
    if not records:
        return ""

    if client is None:
        client = _make_client()

    if client is not None:
        return _llm_compress(records, client)
    else:
        return _rule_compress(records)


def _llm_compress(records: list, client) -> str:
    """Use LLM to compress records into a paragraph."""
    # Build a compact input to reduce token cost
    items = []
    for r in records:
        items.append(
            f"[iter{r.get('iter', '?')}] outcome={r.get('outcome', '?')} "
            f"delta={r.get('delta', 0):+.4f} | "
            f"action: {r.get('action', '')[:120]}"
        )

    prompt = (
        "Compress the following optimization iteration records into concise bullet points "
        "covering: (1) what approaches were tried, (2) what worked and what failed, "
        "(3) any patterns observed. Be specific. Max 150 words.\n\n"
        + "\n".join(items)
    )

    try:
        resp = client.chat.completions.create(
            model=COMPRESS_MODEL["name"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=COMPRESS_MODEL["max_tokens"],
            temperature=COMPRESS_MODEL["temperature"],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        # Fallback on API error
        return _rule_compress(records)


def _rule_compress(records: list) -> str:
    """Rule-based fallback compression (no LLM required)."""
    lines = []
    improved = [r for r in records if r.get("outcome") == "improved"]
    failed   = [r for r in records if r.get("outcome") != "improved"]

    if improved:
        lines.append(
            "Improvements: "
            + "; ".join(
                f"iter{r.get('iter')} delta={r.get('delta', 0):+.4f} [{r.get('action', '')[:60]}]"
                for r in improved
            )
        )
    if failed:
        lines.append(
            "Failed attempts: "
            + "; ".join(
                f"iter{r.get('iter')} [{r.get('action', '')[:60]}]"
                for r in failed[-5:]
            )
        )
    return "\n".join(lines)


class ContextCompressor:
    """
    Manages hot→warm compression for State.

    Call maybe_compress() before building each LLM prompt.
    """

    def __init__(self, state: State, client=None):
        self.state  = state
        self.client = client or _make_client()

    def token_est(self, text: str) -> int:
        """Rough token estimate."""
        zh = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        return int(zh / 1.5 + (len(text) - zh) / 4)

    def hot_token_count(self) -> int:
        hot = self.state.get("hot_history", [])
        return self.token_est(json.dumps(hot))

    def should_compress(self) -> bool:
        threshold = int(CONTEXT["window"] * CONTEXT["compress_threshold"])
        return self.hot_token_count() > threshold

    def compress(self):
        """Move oldest hot records to warm summary."""
        hot = self.state.get("hot_history", [])
        keep_n = CONTEXT["max_hot_iters"]

        if len(hot) <= keep_n:
            return  # nothing to do

        to_compress = hot[:-keep_n]
        compressed_new = compress_iterations(to_compress, self.client)

        existing_warm = self.state.get("warm_summary", "")
        combined = (existing_warm + "\n\n" + compressed_new).strip()

        # Trim warm summary if it grows too large
        max_chars = CONTEXT["max_compressed_chars"]
        if len(combined) > max_chars:
            combined = combined[-max_chars:]

        self.state.update({
            "warm_summary": combined,
            "hot_history":  hot[-keep_n:],
        })

    def maybe_compress(self):
        """Compress if threshold exceeded."""
        if self.should_compress():
            self.compress()
