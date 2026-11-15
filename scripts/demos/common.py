"""Shared pieces for the demo GIF scripts in this folder.

A demo stages one world (a clean stage with one agent and the ghost spectator),
scripts the agent, and records what the spectator sees to ``docs/media/``.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from PIL import Image

from agent_world import CameraConfig, WorldConfig, make_env

MEDIA = Path("docs/media")


def make_demo_env(seed: int = 0, device: str | None = None, **overrides):
    """One world with a clean stage: no obstacles, food, chasers, blocks or patch unless asked."""
    config = dict(
        n_agents=1, spectator=True, n_obstacles=0, food_per_room=0, chasers_per_room=0,
        blocks_per_room=0, rotating_patches=False, camera=CameraConfig(width=192, height=144),
    )
    config.update(overrides)
    env = make_env(num_envs=1, device=device, seed=seed, config=WorldConfig(**config))
    env.reset()
    return env


def put(env, entity, x: float, y: float, heading: float = 0.0) -> None:
    """Place an entity (agent, ghost, chaser or block) and stop it."""
    dev = env.device
    entity.set_pos(torch.tensor([[x, y]], device=dev), batch_index=None)
    entity.set_rot(torch.full((1, 1), heading, device=dev), batch_index=None)
    entity.set_vel(torch.zeros(1, 2, device=dev), batch_index=None)


def look_from(env, ghost, camera_xy, target_xy) -> None:
    """Put the spectator at ``camera_xy`` facing ``target_xy``."""
    dx, dy = target_xy[0] - camera_xy[0], target_xy[1] - camera_xy[1]
    put(env, ghost, camera_xy[0], camera_xy[1], math.atan2(dy, dx))


def follow(env, ghost, a_xy, b_xy, side: float = 1.8) -> None:
    """Stand ``side`` units to the left of the line from a to b, facing its middle.

    The ghost cannot stand outside the room, so keep the action at least ``side`` units
    inside the left wall of the line's direction of travel.
    """
    a = torch.as_tensor(a_xy, dtype=torch.float32).flatten()[:2]
    b = torch.as_tensor(b_xy, dtype=torch.float32).flatten()[:2]
    mid, ahead = (a + b) / 2, b - a
    ahead = ahead / ahead.norm().clamp(min=1e-6)
    left = torch.stack([-ahead[1], ahead[0]])
    camera = mid + side * left
    look_from(env, ghost, camera.tolist(), mid.tolist())


def to_image(rgb: torch.Tensor, scale: int = 3) -> Image.Image:
    array = (rgb.clamp(0, 1) * 255).byte().cpu().numpy()
    image = Image.fromarray(array)
    return image.resize((image.width * scale, image.height * scale), Image.NEAREST)


def save_gif(frames: list[Image.Image], name: str, seconds_per_frame: float = 0.2) -> Path:
    """Write ``docs/media/<name>.gif``; one frame every 2 steps (0.2 s) plays in real time."""
    MEDIA.mkdir(parents=True, exist_ok=True)
    path = MEDIA / f"{name}.gif"
    frames[0].save(
        path, save_all=True, append_images=frames[1:], duration=int(seconds_per_frame * 1000),
        loop=0, optimize=True,
    )
    print(f"Saved {len(frames)} frames to {path} ({path.stat().st_size // 1024} KB)")
    return path
