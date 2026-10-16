import json
import subprocess
from datetime import datetime
from pathlib import Path

from chessbot import SP_DIR

CHESSBOT_DATA_DIR = Path(SP_DIR).resolve().parent
SPEED_TESTING_DIR = CHESSBOT_DATA_DIR / "speed_testing"
FIXTURES_DIR = SPEED_TESTING_DIR / "fixtures"
RUNS_DIR = SPEED_TESTING_DIR / "runs"

CHESSBOT_REPO = Path(__file__).resolve().parents[3]
PYFASTCHESS_REPO = CHESSBOT_REPO.parent / "pyfastchess"


def git_stamp(repo_path):
    def run(args):
        out = subprocess.run(
            args, cwd=repo_path, capture_output=True, text=True, check=False
        )
        return out.stdout.strip()

    sha = run(["git", "rev-parse", "HEAD"])
    dirty = bool(run(["git", "status", "--porcelain"]))
    return {"sha": sha, "dirty": dirty}


def repo_stamps():
    return {
        "chessbot": git_stamp(CHESSBOT_REPO),
        "pyfastchess": git_stamp(PYFASTCHESS_REPO),
    }


def new_run_dir(name):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
