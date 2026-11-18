"""The floor: one large precomputed map, and a randomly placed window into it per world.

The map is circles of three sizes (big, small, fine dots) with rotated rectangle sections laid
on top. Sections are placed one after another, each in the free area outside every earlier
section's bounding box, until the boxes cover a target share of the map. A world's floor is a
randomly placed and rotated window, so nothing on it points the same way in every episode and
there is no global frame to read.

Pure tensor maths apart from drawing the map once with Pillow; knows nothing about VMAS.
"""

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor

SIDE = 62.0
"""Side of the map in world units: about 30 times the area of a 16 by 8 arena."""
TEXEL = 0.03
SIZE = int(SIDE / TEXEL)
ARENA_AREA = 128.0
"""The 16 by 8 arena that all sizes and counts below are relative to."""
HUE = 0.28
SECTION_HUE = HUE + 0.10


def _hsv(h: float, s: float, v: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h % 1, min(max(s, 0), 1), min(max(v, 0), 1))
    return int(255 * r), int(255 * g), int(255 * b)


def place_sections(seed: int, coverage: float = 0.45) -> tuple[list[tuple], float]:
    """Rotated rectangles as (cx, cy, width, height, angle), and the share of the map their
    bounding boxes cover. Each box is placed in the free area, outside all earlier boxes."""
    rng = np.random.default_rng((seed, 1))
    cell = 0.25
    cells = int(SIDE / cell)
    taken = np.zeros((cells, cells), bool)
    sections, covered = [], 0.0
    for _ in range(400_000):
        if covered >= coverage * SIDE * SIDE:
            break
        area = rng.uniform(0.03, 0.10) * ARENA_AREA
        width = math.sqrt(area * rng.uniform(0.5, 2.0))
        height = area / width
        angle = rng.uniform(0, math.pi)
        box_w = width * abs(math.cos(angle)) + height * abs(math.sin(angle))
        box_h = width * abs(math.sin(angle)) + height * abs(math.cos(angle))
        gx, gy = rng.integers(0, cells, 2)
        if taken[gy, gx]:
            continue
        cx, cy = (gx + 0.5) * cell, (gy + 0.5) * cell
        if cx - box_w / 2 < 0 or cy - box_h / 2 < 0 or cx + box_w / 2 > SIDE or cy + box_h / 2 > SIDE:
            continue
        i0, i1 = int((cx - box_w / 2) / cell), math.ceil((cx + box_w / 2) / cell)
        j0, j1 = int((cy - box_h / 2) / cell), math.ceil((cy + box_h / 2) / cell)
        if taken[j0:j1, i0:i1].any():
            continue
        taken[j0:j1, i0:i1] = True
        covered += box_w * box_h
        sections.append((cx, cy, width, height, angle))
    return sections, covered / (SIDE * SIDE)


@lru_cache(maxsize=4)
def build_map(seed: int = 0, coverage: float = 0.45) -> Tensor:
    """The map as uint8 [SIZE, SIZE, 3]. Deterministic in (seed, coverage)."""
    rng = np.random.default_rng((seed, 0))
    image = Image.new("RGB", (SIZE, SIZE), _hsv(HUE, 0.16, 0.68))
    draw = ImageDraw.Draw(image)

    def circles(n, r_lo, r_hi, outline):
        for _ in range(n):
            x, y = rng.uniform(0, SIDE, 2)
            r = rng.uniform(r_lo, r_hi)
            h, s, v = HUE + rng.uniform(-0.03, 0.03), rng.uniform(0.10, 0.32), rng.uniform(0.52, 0.88)
            line = _hsv(h, s + 0.10, v - 0.20) if outline else None
            draw.ellipse(
                [(x - r) / TEXEL, (y - r) / TEXEL, (x + r) / TEXEL, (y + r) / TEXEL],
                fill=_hsv(h, s, v), outline=line, width=max(1, int(outline / TEXEL)) if outline else 0,
            )

    def rectangle(cx, cy, w, h, angle, fill, line, width):
        c, s = math.cos(angle), math.sin(angle)
        corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
        points = [((cx + dx * c - dy * s) / TEXEL, (cy + dx * s + dy * c) / TEXEL) for dx, dy in corners]
        if fill is not None:
            draw.polygon(points, fill=fill)
        draw.line(points + [points[0]], fill=line, width=max(1, int(width / TEXEL)), joint="curve")

    per_arena = SIDE * SIDE / ARENA_AREA
    circles(int(24 * per_arena), 0.8, 2.0, 0.07)
    circles(int(140 * per_arena), 0.12, 0.35, 0.04)
    circles(int(1500 * per_arena), 0.04, 0.10, 0.0)
    inner = np.random.default_rng((seed, 2))
    for cx, cy, w, h, angle in place_sections(seed, coverage)[0]:
        rectangle(cx, cy, w, h, angle, _hsv(SECTION_HUE, 0.18, 0.62), _hsv(SECTION_HUE, 0.28, 0.42), 0.09)
        c, s = math.cos(angle), math.sin(angle)
        for _ in range(int(inner.integers(3, 6))):
            ox, oy = inner.uniform(-w / 2, w / 2), inner.uniform(-h / 2, h / 2)
            hollow = inner.random() < 0.4
            fill = None if hollow else _hsv(
                SECTION_HUE + inner.uniform(-0.03, 0.03), inner.uniform(0.10, 0.32), inner.uniform(0.50, 0.85)
            )
            rectangle(
                cx + ox * c - oy * s, cy + ox * s + oy * c, inner.uniform(0.2, 0.55) * w,
                inner.uniform(0.2, 0.55) * h, inner.uniform(0, math.pi), fill,
                _hsv(SECTION_HUE, 0.30, 0.34), 0.10 if hollow else 0.06,
            )
    return torch.from_numpy(np.asarray(image).copy())


@dataclass
class Floor:
    """One map shared by every world, and each world's window into it."""

    map: Tensor  # [SIZE, SIZE, 3] uint8
    offset: Tensor  # [B, 2] where each world's window is centred in the map
    theta: Tensor  # [B] rotation of each window
    reach: float  # the farthest a floor point is from the arena's centre

    @classmethod
    def create(cls, batch: int, device, reach: float, seed: int = 0, coverage: float = 0.45) -> "Floor":
        if reach >= SIDE / 2:
            raise ValueError(f"the arena (reach {reach}) does not fit in the {SIDE} by {SIDE} floor map")
        floor = cls(build_map(seed, coverage).to(device), torch.zeros(batch, 2, device=device), torch.zeros(batch, device=device), reach)
        floor.reseed()
        return floor

    def reseed(self, env_index: int | None = None) -> None:
        """Move one world's window (or all of them) to a new random place and rotation."""
        rows = slice(None) if env_index is None else env_index
        n = self.offset.shape[0] if env_index is None else 1
        low, high = self.reach, SIDE - self.reach
        self.offset[rows] = (low + torch.rand(n, 2, device=self.offset.device) * (high - low)).reshape(self.offset[rows].shape)
        self.theta[rows] = (torch.rand(n, device=self.theta.device) * 2 * math.pi).reshape(self.theta[rows].shape)

    def colour(self, xy: Tensor) -> Tensor:
        """Floor colour in [0, 1] at the [B, P, 2] arena points."""
        c, s = torch.cos(self.theta)[:, None], torch.sin(self.theta)[:, None]
        x = c * xy[..., 0] - s * xy[..., 1] + self.offset[:, None, 0]
        y = s * xy[..., 0] + c * xy[..., 1] + self.offset[:, None, 1]
        ix = (x / TEXEL).long().clamp(0, SIZE - 1)
        iy = (y / TEXEL).long().clamp(0, SIZE - 1)
        return self.map[iy, ix].float() / 255.0
