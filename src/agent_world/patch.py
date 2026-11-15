"""A floor patch that spins agents, and how it is painted.

The patch is a disc. An agent inside it is turned at ``rate`` radians per
second (negative is clockwise). Painted from above it is a pink disc with a
black ring and black arrowheads along the ring, pointing the way it spins.

Pure tensor maths; knows nothing about VMAS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

RGB = tuple[float, float, float]


@dataclass(frozen=True)
class Patch:
    center: tuple[float, float]
    radius: float = 1.0
    rate: float = -1.0
    """Turn rate added inside the patch, in radians per second. Negative is clockwise."""
    ring_fraction: float = 0.6
    """Radius of the black ring, as a fraction of the patch radius."""
    ring_width: float = 0.06
    n_arrows: int = 4
    """Arrowheads evenly spaced around the ring."""
    arrow_length: float = 0.6
    arrow_width: float = 0.6
    color: RGB = (0.95, 0.55, 0.70)
    ink: RGB = (0.0, 0.0, 0.0)


def inside(pos: Tensor, patch: Patch) -> Tensor:
    """Which of the [..., 2] positions are on the patch."""
    return ((pos - pos.new_tensor(patch.center)) ** 2).sum(-1) <= patch.radius**2


def paint(xy: Tensor, patch: Patch, base: Tensor) -> Tensor:
    """Colour of the ground at the [..., 2] points ``xy``; ``base`` is [..., 3]."""
    d = xy - xy.new_tensor(patch.center)
    r = d.norm(dim=-1)
    phi = torch.atan2(d[..., 1], d[..., 0])
    ring_radius = patch.radius * patch.ring_fraction
    on_ring = (r - ring_radius).abs() <= patch.ring_width / 2

    # Position relative to the nearest arrowhead, in its own straight frame:
    # u along the spin direction (tangent to the ring), v radially outwards.
    step = 2 * math.pi / patch.n_arrows
    delta = torch.remainder(phi + step / 2, step) - step / 2
    spin = 1.0 if patch.rate >= 0 else -1.0
    cos_k, sin_k = torch.cos(phi - delta), torch.sin(phi - delta)
    rel_x, rel_y = d[..., 0] - ring_radius * cos_k, d[..., 1] - ring_radius * sin_k
    u = spin * (-sin_k * rel_x + cos_k * rel_y)
    v = cos_k * rel_x + sin_k * rel_y
    half = patch.arrow_length / 2
    on_arrow = (u.abs() <= half) & (v.abs() <= patch.arrow_width / 2 * (half - u) / patch.arrow_length)

    disc = r <= patch.radius
    color = base.new_tensor(patch.color).expand_as(base)
    ink = base.new_tensor(patch.ink).expand_as(base)
    painted = torch.where((disc & (on_ring | on_arrow))[..., None], ink, color)
    return torch.where(disc[..., None], painted, base)
