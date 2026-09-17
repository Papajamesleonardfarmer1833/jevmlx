"""
Engine router. Apple Silicon (MLX) only — the PyTorch/CUDA backend was removed.
Importing on any other platform fails with a clear message (see engine_mlx).
"""

from openjev.engine_mlx import (
    load_engine,
    run_naive_generation,
    run_parallel_generation,
    stream_naive_generation,
)

__all__ = [
    "load_engine",
    "run_parallel_generation",
    "run_naive_generation",
    "stream_naive_generation",
]
