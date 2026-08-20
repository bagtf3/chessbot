# run_selfplay.py
"""
Entry point. Usage: python run_selfplay.py <run_tag>

Everything lives in chessbot.selfplay_runner, including signal handling and
the operator console, so the mp children import a real module instead of
re-importing this script under spawn.
"""
import sys

from chessbot.selfplay_runner import run_roundless

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_selfplay.py <run_tag>", flush=True)
        sys.exit(1)

    sys.exit(run_roundless(sys.argv[1]))
