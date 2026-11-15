"""Big blue blocks: static solid boxes that make chasers disappear.

Placement is stratified: each room is cut into ``blocks_per_room`` strips along x
and one block sits at a random spot in each, clear of the walls and the doorways.
"""

from __future__ import annotations

import torch
from torch import Tensor

NAME = "block"
BLUE = (0.15, 0.30, 0.85)
TRIES = 16


def place(cfg, n: int, device) -> Tensor:
    """Block centres [n, n_rooms * blocks_per_room, 2] for n worlds."""
    k, half = cfg.blocks_per_room, cfg.block_size / 2
    wall = cfg.wall_thickness / 2 + 0.25 + half
    strip = cfg.room_size / k
    left = -cfg.arena_width / 2 + torch.arange(cfg.n_rooms * k, device=device) * strip
    u = torch.rand(n, cfg.n_rooms * k, TRIES, 2, device=device)
    lo = left + torch.where(torch.arange(cfg.n_rooms * k, device=device) % k == 0, wall, half)
    hi = left + strip - torch.where(torch.arange(cfg.n_rooms * k, device=device) % k == k - 1, wall, half)
    x = lo[:, None] + u[..., 0] * (hi - lo)[:, None]
    y = (u[..., 1] * 2 - 1) * (cfg.room_size / 2 - wall)
    doors = -cfg.arena_width / 2 + torch.arange(1, cfg.n_rooms, device=device) * cfg.room_size
    clear = max(cfg.patch_radius, cfg.doorway_width / 2) + half * 2**0.5 + 0.1
    d = torch.hypot(x[..., None] - doors, y[..., None]).amin(-1) if len(doors) else torch.full_like(x, clear)
    pick = (d >= clear).float().argmax(-1, keepdim=True)
    return torch.stack([x.gather(-1, pick), y.gather(-1, pick)], -1).squeeze(2)


def touching(pos: Tensor, blocks: Tensor, half: float, radius: float) -> Tensor:
    """Which circles [B, C, 2] touch any axis-aligned square block [B, K, 2]: [B, C]."""
    gap = ((pos[:, :, None] - blocks[:, None]).abs() - half).clamp(min=0)
    return (gap.norm(dim=-1) <= radius).any(-1)
