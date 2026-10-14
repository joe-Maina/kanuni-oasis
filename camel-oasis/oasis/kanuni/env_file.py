import os
from pathlib import Path

ENV_PATH = Path(__file__).parent / ".env"


def load_env_file(path: Path = ENV_PATH) -> None:
    """Load KEY=VALUE lines from kanuni/.env without overriding the shell."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
