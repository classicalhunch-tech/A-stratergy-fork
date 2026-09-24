"""
ai_debug_wrapper.py

Runs your bot script (default: main.py) as a subprocess, watches its
stderr in real time, and if a Python traceback appears, sends it to a
local Ollama model for a plain-English explanation + suggested fix.

Usage:
    python ai_debug_wrapper.py                 # runs main.py
    python ai_debug_wrapper.py dashboard/app.py # runs a different script
    python ai_debug_wrapper.py main.py --model qwen2.5-coder:3b

Requirements:
    - Ollama installed and running (it starts automatically as a service
      after `ollama pull`, but if not, run `ollama serve` in another window)
    - requests installed: pip install requests --break-system-packages
      (or just: pip install requests, inside your .venv)
"""

import argparse
import subprocess
import sys
import re
import json
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"


def ask_ollama(traceback_text: str, model: str) -> str:
    """Send a traceback to the local Ollama model and return its explanation."""
    prompt = (
        "You are a terse Python debugging assistant. A traceback from a "
        "trading bot script is below. In 4-6 short lines, state:\n"
        "1) what error this is\n"
        "2) the most likely cause given the file/line shown\n"
        "3) a concrete suggested fix\n"
        "Do not repeat the full traceback back to me.\n\n"
        f"TRACEBACK:\n{traceback_text}\n"
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    try:
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()
    except Exception as e:
        return f"[ai_debug_wrapper] Could not reach Ollama ({e}). Is it running? Try: ollama serve"


def run_and_watch(script_path: str, model: str, extra_args: list[str]):
    cmd = [sys.executable, script_path] + extra_args
    print(f"[ai_debug_wrapper] Launching: {' '.join(cmd)}")
    print(f"[ai_debug_wrapper] Using model: {model}\n")

    proc = subprocess.Popen(
        cmd,
        stderr=subprocess.PIPE,
        stdout=sys.stdout,
        text=True,
        bufsize=1,
    )

    traceback_lines = []
    in_traceback = False

    assert proc.stderr is not None
    for line in proc.stderr:
        sys.stderr.write(line)  # always show the raw output too
        sys.stderr.flush()

        if line.startswith("Traceback (most recent call last):"):
            in_traceback = True
            traceback_lines = [line]
            continue

        if in_traceback:
            traceback_lines.append(line)
            # A traceback block ends on the line with the actual exception,
            # e.g. "IndexError: list index out of range" (no leading whitespace)
            if re.match(r"^\S+Error", line) or re.match(r"^\S+Exception", line):
                in_traceback = False
                full_tb = "".join(traceback_lines)
                print("\n[ai_debug_wrapper] Traceback detected — asking local model...\n")
                explanation = ask_ollama(full_tb, model)
                print("=" * 60)
                print(explanation)
                print("=" * 60 + "\n")

    proc.wait()
    print(f"\n[ai_debug_wrapper] Script exited with code {proc.returncode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a script and auto-explain any tracebacks via local Ollama.")
    parser.add_argument("script", nargs="?", default="main.py", help="Script to run (default: main.py)")
    parser.add_argument("--model", default="qwen2.5-coder:3b", help="Ollama model to use")
    args, unknown = parser.parse_known_args()

    run_and_watch(args.script, args.model, unknown)