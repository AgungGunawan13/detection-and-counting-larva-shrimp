import time
from pathlib import Path
import os
from pathlib import Path

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def print_header(title: str):
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")

def find_yaml(root_dir: str | Path):
    root = Path(root_dir)
    yamls = list(root.rglob("*.yaml")) + list(root.rglob("*.yml"))
    return yamls[0] if yamls else None

# Root folder project (2 level di atas file ini)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def chdir_project_root():
    os.chdir(PROJECT_ROOT)