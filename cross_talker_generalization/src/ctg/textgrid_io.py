from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import MutableMapping


_ITEM_RE = re.compile(r"^\s*item\s*\[\d+\]\s*:\s*$")
_INTERVAL_RE = re.compile(r"^\s*intervals\s*\[(?P<index>\d+)\]\s*:\s*$")
_NAME_RE = re.compile(r'^\s*name\s*=\s*"(?P<value>.*)"\s*$')
_XMIN_RE = re.compile(r"^\s*xmin\s*=\s*(?P<value>[-+0-9.eE]+)\s*$")
_XMAX_RE = re.compile(r"^\s*xmax\s*=\s*(?P<value>[-+0-9.eE]+)\s*$")
_TEXT_RE = re.compile(r'^\s*text\s*=\s*"(?P<value>.*)"\s*$')


@dataclass(frozen=True)
class Interval:
    index: int
    xmin: float
    xmax: float
    text: str


def parse_long_textgrid(path: str | Path) -> dict[str, list[Interval]]:
    """Parse the repository's long-form Praat TextGrids without gap repair."""

    path = Path(path)
    tiers: dict[str, list[Interval]] = {}
    tier_name = None
    intervals: list[Interval] = []
    current: MutableMapping[str, object] | None = None
    saw_item = False

    def finish_interval():
        nonlocal current
        if current is None:
            return
        missing = {"index", "xmin", "xmax", "text"}.difference(current)
        if missing:
            raise ValueError(f"incomplete TextGrid interval in {path}: {sorted(missing)}")
        intervals.append(
            Interval(
                int(current["index"]),
                float(current["xmin"]),
                float(current["xmax"]),
                str(current["text"]),
            )
        )
        current = None

    def finish_tier():
        nonlocal tier_name, intervals
        finish_interval()
        if tier_name is not None:
            if tier_name in tiers:
                raise ValueError(f"duplicate TextGrid tier {tier_name!r}")
            tiers[tier_name] = intervals
        tier_name = None
        intervals = []

    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if _ITEM_RE.match(line):
                if saw_item:
                    finish_tier()
                saw_item = True
                continue
            if not saw_item:
                continue
            match = _NAME_RE.match(line)
            if match and current is None:
                tier_name = match.group("value").replace('""', '"')
                continue
            match = _INTERVAL_RE.match(line)
            if match:
                finish_interval()
                current = {"index": int(match.group("index"))}
                continue
            if current is None:
                continue
            for key, pattern in (("xmin", _XMIN_RE), ("xmax", _XMAX_RE), ("text", _TEXT_RE)):
                match = pattern.match(line)
                if match:
                    value: object = match.group("value")
                    current[key] = value.replace('""', '"') if key == "text" else float(value)
                    break
    if saw_item:
        finish_tier()
    if not tiers:
        raise ValueError(f"no interval tiers found in {path}")
    return tiers
