from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = ROOT / "configs" / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def p(rel: str) -> Path:
    """Resolve a repo-relative path."""
    return ROOT / rel
