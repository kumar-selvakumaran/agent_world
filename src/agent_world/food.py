"""Floor food: small flat discs that agents eat, and how they are painted.

Food is not an entity: each world holds F positions [B, F, 2] and a timer
[B, F]. A timer of 0 means the food is there; above 0 it has been eaten and
reappears at a new random spot in its own room when the timer runs out.
Food ``i`` belongs to room ``i // per_room``.

Pure tensor maths; knows nothing about VMAS.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

RGB = tuple[float, float, float]
COLOR: RGB = (0.80, 0.58, 0.12)


@dataclass
class Food:
    """Snapshot for painting."""

    xy: Tensor  # [B, F, 2]
    active: Tensor  # [B, F] bool
    radius: float


def sample(n: int, per_room: int, n_rooms: int, room_size: float, margin: float, device) -> Tensor:
    """Uniform positions [n, F, 2] inside each room, ``margin`` from its walls; arena centred on 0."""
    room = torch.arange(n_rooms * per_room, device=device) // per_room
    left = -n_rooms * room_size / 2 + room * room_size  # [F]
    u = torch.rand(n, n_rooms * per_room, 2, device=device)
    span = room_size - 2 * margin
    x = left + margin + u[..., 0] * span
    y = -room_size / 2 + margin + u[..., 1] * span
    return torch.stack((x, y), dim=-1)


def step(
    xy: Tensor, timer: Tensor, agent_pos: Tensor, can_eat: Tensor, reach: float, respawn: float, dt: float, new_xy: Tensor
) -> None:
    """Advance one step in place: tick timers, respawn the due food at ``new_xy``, then eat.

    agent_pos is [B, A, 2]; can_eat is [A] bool; eating needs an agent centre within ``reach``.
    """
    waiting = timer > 0
    timer -= dt * waiting
    back = waiting & (timer <= 1e-4)
    timer[back] = 0.0
    xy[back] = new_xy[back]
    near = ((agent_pos[:, can_eat, None] - xy[:, None]) ** 2).sum(-1) <= reach**2  # [B, A, F]
    timer[near.any(1) & (timer == 0)] = respawn


def paint(xy: Tensor, food: Food, base: Tensor, color: RGB = COLOR) -> Tensor:
    """Colour of the ground at the [..., 2] points ``xy``; ``base`` is [..., 3]. One food at a time keeps memory small."""
    flat = xy.reshape(xy.shape[0], -1, 2)  # [B, P, 2]
    hit = torch.zeros(flat.shape[:2], dtype=torch.bool, device=xy.device)
    for f in range(food.xy.shape[1]):
        hit |= (((flat - food.xy[:, None, f]) ** 2).sum(-1) <= food.radius**2) & food.active[:, None, f]
    return torch.where(hit.reshape(xy.shape[:-1])[..., None], base.new_tensor(color).expand_as(base), base)
