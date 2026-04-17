from __future__ import annotations
from typing import List
from pathlib import Path

def read_lines(path: Path, max_lines: int = 0) -> List[str]:
    lines: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
            if max_lines and len(lines) >= max_lines:
                break
    return lines

def chunk_lines(lines: List[str], chunk_size: int) -> List[List[str]]:
    if chunk_size <= 0:
        return [lines]
    return [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
