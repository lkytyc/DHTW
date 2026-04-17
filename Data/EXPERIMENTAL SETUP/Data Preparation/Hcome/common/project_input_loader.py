from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml

def load_project_input(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("project_input.yaml must be a YAML mapping.")
    for key in ["domain", "requirements", "output"]:
        if key not in data:
            raise ValueError(f"Missing required section '{key}' in project_input.yaml")
    return data
