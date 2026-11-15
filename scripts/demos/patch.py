"""Record the rotating patch: a spin with no steering, and a crossing that counter-steers.

    uv run python scripts/demos/patch.py

Writes ``docs/media/patch_spin.gif`` and ``docs/media/patch_cross.gif``.
"""

from __future__ import annotations

import torch

from common import look_from, make_demo_env, put, save_gif, to_image


def record(name, start, command_at, camera, target, steps, scale=2):
    env = make_demo_env(rotating_patches=True)
    agent, ghost = env.agents
    put(env, agent, *start)
    look_from(env, ghost, camera(agent.state.pos[0]), target(agent.state.pos[0]))
    print(f"{name}: camera at {ghost.state.pos[0].tolist()}")
    frames = []
    for step in range(steps):
        look_from(env, ghost, camera(agent.state.pos[0]), target(agent.state.pos[0]))
        command = command_at(step * 0.1, agent.state.pos[0], agent.state.rot[0, 0].item())
        actions = [torch.tensor([command], device=env.device), torch.zeros(1, 2, device=env.device)]
        obs, _, _, _ = env.step(actions)
        if step % 2 == 0:
            frames.append(to_image(obs[-1][0], scale))
    print(f"{name}: agent ends at {agent.state.pos[0].tolist()}, heading {agent.state.rot[0, 0].item():.2f}")
    save_gif(frames, name)


def spin(t, pos, heading):
    return [1.0, 0.0] if t < 1.8 else [0.0, 0.0]


def cross(t, pos, heading):
    return [1.0, 0.5] if t < 8.0 else [0.0, 0.0]


if __name__ == "__main__":
    record("patch_spin", (-2.2, 0.0, 0.0), spin, lambda p: (-0.5, -0.9), lambda p: (-0.3, 0.0), 100)
    record("patch_cross", (-0.95, -0.9, 0.0), cross, lambda p: (-1.5, -1.7), lambda p: (0.5, -0.2), 80)
