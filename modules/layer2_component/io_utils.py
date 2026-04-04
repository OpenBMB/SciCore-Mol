from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterator, TextIO


def open_text(path: str | Path, mode: str) -> TextIO:
 """
 .gz file
 - mode supportsmode'r'/'w'/'a' 
 """
 p = Path(path)
 if "b" in mode:
 raise ValueError("open_text supportsmode 'b'")
 if p.suffix == ".gz":
 return gzip.open(p, mode + "t", encoding="utf-8") # type: ignore[return-value]
 return p.open(mode, encoding="utf-8")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
 with open_text(path, "r") as f:
 for line in f:
 line = line.strip()
 if not line:
 continue
 yield json.loads(line)


def write_jsonl_line(f: TextIO, obj: dict[str, Any]) -> None:
 f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")

