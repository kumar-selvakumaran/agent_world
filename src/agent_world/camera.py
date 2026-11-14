"""Third-person camera: turns an agent pose into a bundle of viewing rays.

The eye sits ``eye_height`` above the ground, at (or ``eye_back`` behind) the
agent, and looks along the agent's heading, tilted down. The view is bounded

* at the top by the horizon,
* at the bottom by the ground point ``rear_distance`` behind the agent,
* left and right by ``half_fov_deg``.

It is an ordinary perspective (pinhole) camera, so straight lines stay
straight. Pitch is measured below the horizon: 0 is the horizon ahead, 90 deg
is straight down, and more than 90 deg looks backwards - which is what brings
the agent's own body into view. The optical axis is pitched halfway between
the horizon and the bottom edge.

This module is pure tensor maths and knows nothing about VMAS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CameraConfig:
    eye_height: float = 0.6
    """Height of the eye above the ground. Should exceed the agent's height."""
    eye_back: float = 0.0
    """How far behind the agent's centre the eye sits (0 = directly above)."""
    rear_distance: float = 0.2
    """The bottom edge of the view meets the ground this far behind the agent."""
    half_fov_deg: float = 60.0
    """Horizontal half field of view: the view spans +-this angle."""
    width: int = 64
    height: int = 48
    max_range: float = 12.0
    """Anything farther than this is drawn as background."""

    @property
    def phi_max(self) -> float:
        """Pitch of the bottom edge of the view, in radians."""
        return math.pi / 2 + math.atan2(
            self.rear_distance - self.eye_back, self.eye_height
        )


def local_ray_directions(cfg: CameraConfig, device=None) -> Tensor:
    """Unit ray directions in the agent frame (forward, left, up).

    Returns a tensor of shape [height * width, 3], row-major, with the top-left
    pixel first. Pixel centres are used, so no ray is exactly horizontal.
    """
    pitch = cfg.phi_max / 2  # optical axis, below the horizon
    axis = torch.tensor([math.cos(pitch), 0.0, -math.sin(pitch)], device=device)
    up = torch.tensor([math.sin(pitch), 0.0, math.cos(pitch)], device=device)
    left = torch.tensor([0.0, 1.0, 0.0], device=device)

    # Image-plane coordinates at unit distance along the axis.
    rows = (torch.arange(cfg.height, device=device) + 0.5) / cfg.height
    cols = (torch.arange(cfg.width, device=device) + 0.5) / cfg.width
    v = (1 - 2 * rows) * math.tan(pitch)  # top row ~ horizon
    u = (1 - 2 * cols) * math.tan(math.radians(cfg.half_fov_deg))  # left column first
    v, u = torch.meshgrid(v, u, indexing="ij")

    dirs = axis + u[..., None] * left + v[..., None] * up
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    return dirs.reshape(-1, 3)


def world_rays(
    cfg: CameraConfig, local_dirs: Tensor, pos: Tensor, heading: Tensor
) -> tuple[Tensor, Tensor]:
    """Place the camera on an agent.

    Args:
        local_dirs: [P, 3] from :func:`local_ray_directions`.
        pos: [B, 2] agent positions.
        heading: [B] agent headings in radians.

    Returns:
        origin [B, 3] and directions [B, P, 3], both in world coordinates.
    """
    cos, sin = torch.cos(heading), torch.sin(heading)
    forward = torch.stack((cos, sin), dim=-1)  # [B, 2]
    eye_xy = pos - cfg.eye_back * forward
    eye_z = torch.full_like(heading, cfg.eye_height).unsqueeze(-1)
    origin = torch.cat((eye_xy, eye_z), dim=-1)

    f, l, u = local_dirs.unbind(-1)  # each [P]
    cos, sin = cos.unsqueeze(-1), sin.unsqueeze(-1)  # [B, 1]
    dirs = torch.stack(
        (f * cos - l * sin, f * sin + l * cos, u.expand(len(heading), -1)), dim=-1
    )
    return origin, dirs
