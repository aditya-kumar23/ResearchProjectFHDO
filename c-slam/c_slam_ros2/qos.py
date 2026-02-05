from __future__ import annotations

from typing import Dict, Optional

DEFAULT_RELIABILITY = "reliable"
DEFAULT_DURABILITY = "volatile"
DEFAULT_DEPTH = 10


def default_qos_profile() -> Dict[str, object]:
    return {"reliability": DEFAULT_RELIABILITY, "durability": DEFAULT_DURABILITY, "depth": DEFAULT_DEPTH}


_VALID_RELIABILITY = {"reliable", "best_effort"}
_VALID_DURABILITY = {"volatile", "transient_local"}


def parse_qos_options(
    *,
    reliability: Optional[str] = None,
    durability: Optional[str] = None,
    depth: Optional[int] = None,
) -> Dict[str, object]:
    profile = default_qos_profile()
    if reliability:
        norm = reliability.lower()
        if norm not in _VALID_RELIABILITY:
            raise ValueError(f"Unsupported reliability policy {reliability!r}")
        profile["reliability"] = norm
    if durability:
        norm = durability.lower()
        if norm not in _VALID_DURABILITY:
            raise ValueError(f"Unsupported durability policy {durability!r}")
        profile["durability"] = norm
    if depth is not None:
        if int(depth) <= 0:
            raise ValueError("QoS depth must be positive")
        profile["depth"] = int(depth)
    return profile

