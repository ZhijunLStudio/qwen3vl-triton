"""
agent/tools.py - Tool schemas in OpenAI function-calling format

Matches the tools implemented in core/executor.py.
"""

TOOLS = [
    # ── File Operations ──────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": (
                "Read a file in the project directory. "
                "Use to inspect evaluation_wrapper.py, benchmark.py, profiling output, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to project root (e.g. 'evaluation_wrapper.py')",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Start reading from this line number (0-indexed). Default 0.",
                        "default": 0,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of lines to return. 0 = all. Default 0.",
                        "default": 0,
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": (
                "Write (create or fully overwrite) evaluation_wrapper.py. "
                "Use this when making major structural changes. "
                "For small targeted changes, prefer 'edit' instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Must be 'evaluation_wrapper.py'",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file content (valid Python)",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": (
                "Make a targeted in-place edit to evaluation_wrapper.py. "
                "Replaces the first occurrence of old_str with new_str. "
                "old_str must match exactly and appear only once. "
                "Prefer this over full 'write' for small changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Must be 'evaluation_wrapper.py'",
                    },
                    "old_str": {
                        "type": "string",
                        "description": (
                            "Exact string to find (must be unique in the file, "
                            "include enough context to be unambiguous)"
                        ),
                    },
                    "new_str": {
                        "type": "string",
                        "description": "Replacement string",
                    },
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },

    # ── Shell ────────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a shell command (runs inside the torch conda env). "
                "Use for profiling, package inspection, GPU status checks, etc. "
                "Do NOT use for benchmark or accuracy — use run_benchmark instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 60, max 300)",
                        "default": 60,
                    },
                },
                "required": ["command"],
            },
        },
    },

    # ── Search ───────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search for a regex pattern in project files. "
                "Useful for finding function definitions, imports, or code patterns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (relative to project root). Default '.'",
                        "default": ".",
                    },
                    "glob": {
                        "type": "string",
                        "description": "File glob filter (e.g. '*.py'). Default '*.py'",
                        "default": "*.py",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Case-sensitive search. Default true.",
                        "default": True,
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "List files matching a glob pattern in the project directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g. '**/*.py', '*.json')",
                    },
                    "path": {
                        "type": "string",
                        "description": "Base directory (relative to project root). Default '.'",
                        "default": ".",
                    },
                },
                "required": ["pattern"],
            },
        },
    },

    # ── Web ──────────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "webfetch",
            "description": (
                "Fetch a web page and return its text content. "
                "Use to look up documentation, PyPI package info, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL to fetch (https://...)",
                    },
                },
                "required": ["url"],
            },
        },
    },

    # ── Optimization-specific ────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "run_benchmark",
            "description": (
                "Run the full benchmark (TTFT + throughput + accuracy). "
                "Score is computed automatically by the system — you will see "
                "ttft_ms, throughput, accuracy, score, delta_score, and is_improvement. "
                "ALWAYS call this after modifying evaluation_wrapper.py."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "commit_if_improved",
            "description": (
                "If the last benchmark showed improvement, commit evaluation_wrapper.py "
                "to git and push. If not improved or accuracy dropped, auto-reverts. "
                "Must call run_benchmark before this."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "revert_changes",
            "description": (
                "Revert evaluation_wrapper.py to the last committed version. "
                "Call this when an optimization did not help or caused errors."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_lesson",
            "description": (
                "Save a lesson to persistent long-term memory. "
                "Call after every attempt (success or failure) to build knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["worked", "failed", "observation", "strategy"],
                        "description": (
                            "worked=confirmed improvement, failed=did not help, "
                            "observation=finding about the model/hw, strategy=general approach"
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "Brief, specific lesson (1-2 sentences)",
                    },
                },
                "required": ["category", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_status",
            "description": (
                "Get the full current optimization state: baseline/current/best metrics, "
                "iteration history, goals, long-term memory. "
                "Call at the start of each iteration to orient yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]
