"""Shared config loading utility for the Erben Blind pipeline.

All pipeline modules load config files through this single function,
which caches results to avoid repeated disk reads.
"""

import json
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

_cache: dict[str, Any] = {}


def load_config(filename: str) -> dict[str, Any]:
    """Load a JSON config file from the config/ directory.

    Caches loaded configs so repeated calls don't re-read disk.

    Args:
        filename: Name of the JSON file (e.g., 'erben_table.json').

    Returns:
        Parsed JSON as a dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if filename in _cache:
        return _cache[filename]

    path = CONFIG_DIR / filename
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    _cache[filename] = data
    return data
