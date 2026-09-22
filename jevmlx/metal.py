"""Shared Metal buffer-cache helpers (issue #79, #149).

The Metal allocator hoards freed buffers and pushes the machine into swap;
``mx.set_cache_limit`` makes the allocator evict buffers above the cap
instead of hoarding, and ``mx.clear_cache`` returns them to macOS.

Both bench (long multi-combo runs, #79) and serve (long-running daemon,
#149) need these. This is the single owner; bench and serve import from
here so the logic is not duplicated.

All helpers are best-effort: a non-Metal build (CPU, CI) returns a safe
default and never raises.
"""

from __future__ import annotations


def set_cache_limit(cache_gb: float) -> int | None:
    """Cap the Metal buffer cache; returns the bytes set, or None.

    ``cache_gb`` is converted to bytes and passed to ``mx.set_cache_limit``.
    Best-effort: a non-Metal build returns None and never raises.
    """
    try:
        import mlx.core as mx

        limit_bytes = int(float(cache_gb) * 2**30)
        mx.set_cache_limit(limit_bytes)
        return limit_bytes
    except Exception:  # noqa: BLE001 - memory config must never break startup
        return None


def clear_cache() -> None:
    """Release the Metal buffer cache; never raises."""
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:  # noqa: BLE001 - cleanup must never break the run
        pass


def cache_memory_bytes() -> int:
    """Metal buffer-cache bytes, or -1 when unreadable."""
    try:
        import mlx.core as mx

        return mx.get_cache_memory()
    except Exception:  # noqa: BLE001 - memory reading must never break the run
        return -1


def cache_memory_gb() -> float:
    """Metal buffer-cache bytes in GB, or -1.0 when unreadable."""
    b = cache_memory_bytes()
    return round(b / 2**30, 2) if b >= 0 else -1.0
