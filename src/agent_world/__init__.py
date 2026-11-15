"""Agent World: a 2.5D third-person world built on VMAS."""

from __future__ import annotations

from agent_world import config  # noqa: F401  (sets download locations first)

import torch

from agent_world.camera import CameraConfig
from agent_world.raycast import Palette
from agent_world.scenario import Scenario, WorldConfig

__all__ = [
    "CameraConfig",
    "Palette",
    "Scenario",
    "WorldConfig",
    "default_device",
    "make_env",
]


def default_device() -> str:
    """The GPU if there is one, otherwise the CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_env(
    num_envs: int = 1,
    device: str | torch.device | None = None,
    config: WorldConfig | None = None,
    **kwargs,
):
    """Create the VMAS environment for this world.

    Physics and rendering both run on ``device``; ``None`` picks the GPU when
    available. Observations come back on that device, and actions should be
    created there too (``env.device``). ``kwargs`` are forwarded to
    :func:`vmas.make_env` (e.g. ``max_steps``, ``seed``).
    """
    import vmas

    return vmas.make_env(
        scenario=Scenario(),
        num_envs=num_envs,
        device=device or default_device(),
        config=config or WorldConfig(),
        **kwargs,
    )
