"""Modular 2D FDTD package for a ring-array transducer."""

from .array import RingArray
from .ct_medium import CTRingGrid, load_ct_ring_grid
from .experiment import (
    axis_centers,
    n_steps_for_crossing,
    simulate_shot,
    stable_dt,
)
from .solver import FDTD2D
from .sources import linear_chirp
from .wave import ScalarWave2D, WaveModel

__all__ = [
    "FDTD2D",
    "CTRingGrid",
    "RingArray",
    "ScalarWave2D",
    "WaveModel",
    "axis_centers",
    "linear_chirp",
    "load_ct_ring_grid",
    "n_steps_for_crossing",
    "simulate_shot",
    "stable_dt",
]
