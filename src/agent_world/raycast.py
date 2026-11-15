"""Ray casting against a 2.5D scene.

The physics is flat, but for vision every entity is extruded upwards from the
ground to a height: circles become cylinders, boxes become prisms. Casting the
camera's rays against these gives perspective, occlusion and the agent's own
body in view from one mechanism.

Pure tensor maths; knows nothing about VMAS.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from agent_world.patch import Patch, paint

_EPS = 1e-6
_INF = float("inf")


@dataclass
class Scene:
    """A batch of worlds, each with the same E entities in different poses."""

    pos: Tensor  # [B, E, 2] centres
    rot: Tensor  # [B, E]    rotation about the vertical axis
    half_size: Tensor  # [E, 2]  box half length/width; cylinders use [:, 0] = radius
    height: Tensor  # [E]
    is_box: Tensor  # [E] bool; False = cylinder
    color: Tensor  # [E, 3] in [0, 1]
    color2: Tensor | None = None  # [E, 3] second colour of striped entities
    stripes: Tensor | None = None  # [E] int; bands across the body along its forward axis, 0 = plain


@dataclass(frozen=True)
class Palette:
    ground_a: tuple[float, float, float] = (0.80, 0.80, 0.78)
    ground_b: tuple[float, float, float] = (0.70, 0.70, 0.68)
    background: tuple[float, float, float] = (0.55, 0.70, 0.90)
    tile: float = 0.5
    """Side of the ground's checkerboard squares."""
    side_shade: float = 0.7
    """Vertical faces are darkened by this factor so shapes read as solid."""


def _safe(d: Tensor) -> Tensor:
    """Nudge near-zero components so divisions stay finite."""
    return torch.where(d.abs() < _EPS, torch.full_like(d, _EPS), d)


def intersect(origin: Tensor, dirs: Tensor, scene: Scene) -> Tensor:
    """Distance along each ray to each entity.

    Args:
        origin: [B, 3] ray origins.
        dirs: [B, P, 3] unit ray directions.

    Returns:
        [B, P, E] distances; ``inf`` where the ray misses.
    """
    # Move rays into each entity's own frame: [B, P, E, 2] horizontally.
    cos, sin = torch.cos(scene.rot), torch.sin(scene.rot)  # [B, E]
    rel = origin[:, None, :2] - scene.pos  # [B, E, 2]
    o_xy = torch.stack(
        (rel[..., 0] * cos + rel[..., 1] * sin, -rel[..., 0] * sin + rel[..., 1] * cos),
        dim=-1,
    )[:, None]  # [B, 1, E, 2]
    dx, dy = dirs[..., None, 0], dirs[..., None, 1]  # [B, P, 1]
    d_xy = torch.stack(
        (dx * cos[:, None] + dy * sin[:, None], -dx * sin[:, None] + dy * cos[:, None]),
        dim=-1,
    )  # [B, P, E, 2]
    o_z = origin[:, None, None, 2]  # [B, 1, 1]
    d_z = _safe(dirs[..., None, 2])  # [B, P, 1]

    # Each shape type is tested only against the entities of that type.
    t = d_xy.new_full(d_xy.shape[:-1], _INF)
    for mask, hit in ((scene.is_box, _hit_prisms), (~scene.is_box, _hit_cylinders)):
        if mask.any():
            t[..., mask] = hit(
                o_xy[:, :, mask],
                d_xy[:, :, mask],
                o_z,
                d_z,
                scene.half_size[mask],
                scene.height[mask],
            )
    return t


def _hit_prisms(o_xy, d_xy, o_z, d_z, half_size, height) -> Tensor:
    """Slab test on x, y and z."""
    d_xy = _safe(d_xy)
    t1 = (-half_size - o_xy) / d_xy
    t2 = (half_size - o_xy) / d_xy
    tz1, tz2 = -o_z / d_z, (height - o_z) / d_z
    t_near = torch.maximum(torch.minimum(t1, t2).amax(-1), torch.minimum(tz1, tz2))
    t_far = torch.minimum(torch.maximum(t1, t2).amin(-1), torch.maximum(tz1, tz2))
    t_near = t_near.clamp(min=0)  # an eye inside the prism sees it at distance 0
    return torch.where(t_far >= t_near, t_near, _INF)


def _hit_cylinders(o_xy, d_xy, o_z, d_z, half_size, height) -> Tensor:
    """Curved side, then the top disc."""
    radius_sq = half_size[:, 0] ** 2
    a = (d_xy**2).sum(-1).clamp(min=_EPS)
    b = (o_xy * d_xy).sum(-1)
    c = (o_xy**2).sum(-1) - radius_sq
    disc = b**2 - a * c
    t_side = (-b - disc.clamp(min=0).sqrt()) / a
    z_side = o_z + t_side * d_z
    side_hit = (disc >= 0) & (t_side > 0) & (z_side >= 0) & (z_side <= height)

    t_top = (height - o_z) / d_z
    top_xy = o_xy + t_top[..., None] * d_xy
    top_hit = (t_top > 0) & ((top_xy**2).sum(-1) <= radius_sq)

    return torch.minimum(
        torch.where(side_hit, t_side, _INF), torch.where(top_hit, t_top, _INF)
    )


def render(
    origin: Tensor,
    dirs: Tensor,
    scene: Scene,
    max_range: float,
    palette: Palette = Palette(),
    visible: Tensor | None = None,
    patches: tuple[Patch, ...] = (),
) -> tuple[Tensor, Tensor]:
    """Colour and depth seen along each ray.

    Args:
        visible: optional [E] bool mask; entities marked False are not drawn.
        patches: floor patches painted over the checkerboard.

    Returns:
        rgb [B, P, 3] in [0, 1] and depth [B, P], clipped to ``max_range``.
    """
    t = intersect(origin, dirs, scene)
    if visible is not None:
        t = torch.where(visible, t, _INF)
    t_entity, idx = t.min(dim=-1)  # [B, P]

    # Ground plane z = 0, drawn as a checkerboard so motion is visible.
    t_ground = -origin[:, None, 2] / _safe(dirs[..., 2])
    t_ground = torch.where(t_ground > 0, t_ground, _INF)
    ground_xy = origin[:, None, :2] + t_ground.nan_to_num(posinf=0.0)[..., None] * dirs[..., :2]
    tiles = torch.floor(ground_xy / palette.tile).sum(-1) % 2 == 0
    ground_rgb = torch.where(
        tiles[..., None],
        origin.new_tensor(palette.ground_a),
        origin.new_tensor(palette.ground_b),
    )

    for patch in patches:
        ground_rgb = paint(ground_xy, patch, ground_rgb)

    # Entities: top faces at full colour, vertical faces darker.
    hit_z = origin[:, None, 2] + t_entity.nan_to_num(posinf=0.0) * dirs[..., 2]
    on_top = hit_z >= scene.height[idx] - 1e-4
    shade = torch.where(on_top, 1.0, palette.side_shade)[..., None]
    entity_color = scene.color[idx]
    if scene.stripes is not None:
        entity_color = _striped(entity_color, origin, dirs, t_entity, idx, scene)
    entity_rgb = entity_color * shade

    depth = torch.minimum(t_entity, t_ground)
    rgb = torch.where((t_entity <= t_ground)[..., None], entity_rgb, ground_rgb)
    rgb = torch.where(
        (depth > max_range)[..., None], origin.new_tensor(palette.background), rgb
    )
    return rgb, depth.clamp(max=max_range)


def _striped(color: Tensor, origin: Tensor, dirs: Tensor, t: Tensor, idx: Tensor, scene: Scene) -> Tensor:
    """Alternate ``color`` and ``color2`` in bands along each hit entity's forward axis."""
    rows = torch.arange(idx.shape[0], device=idx.device)[:, None]
    hit_xy = origin[:, None, :2] + t.nan_to_num(posinf=0.0)[..., None] * dirs[..., :2]
    rel = hit_xy - scene.pos[rows, idx]
    rot = scene.rot[rows, idx]
    along = rel[..., 0] * torch.cos(rot) + rel[..., 1] * torch.sin(rot)
    n = scene.stripes[idx]
    band = torch.floor((along / (2 * scene.half_size[idx, 0]) + 0.5) * n).clamp(min=0).long()
    second = (n > 0) & (band % 2 == 1)
    return torch.where(second[..., None], scene.color2[idx], color)
