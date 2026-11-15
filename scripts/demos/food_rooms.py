"""Record two README demos: an agent eating a line of food, and an agent crossing the doorway.

    uv run python scripts/demos/food_rooms.py

Writes ``docs/media/food_eating.gif`` and ``docs/media/rooms.gif``.
"""

from __future__ import annotations

import torch

from common import look_from, make_demo_env, put, save_gif, to_image


def record(env, steps, speed, start, camera, target):
    agent, ghost = env.agents
    put(env, agent, *start)
    if camera:
        look_from(env, ghost, camera, target)
    frames = []
    for step in range(steps):
        if camera is None:
            x = agent.state.pos[0, 0].item()
            look_from(env, ghost, (x - 2.0, 0.0), (x + 3.0, 0.0))
        obs, _, _, _ = env.step([torch.tensor([[speed, 0.0]], device=env.device), torch.zeros(1, 2, device=env.device)])
        if step % 2 == 0:
            frames.append(to_image(obs[-1][0], 3))
    return frames


def food_eating():
    env = make_demo_env(food_per_room=5)
    xy = env.scenario._food_xy
    xy[0, :5] = torch.tensor([[-4.0 + 0.5 * i, 0.0] for i in range(5)], device=env.device)
    xy[0, 5:] = torch.tensor([[6.0, 3.0]] * 5, device=env.device)
    env.scenario._food_timer.zero_()
    print("food placed:", xy[0, :5].tolist())
    frames = record(env, 38, 0.8, (-4.6, 0.0), (-3.0, 0.9), (-3.0, 0.0))
    save_gif(frames, "food_eating")


def rooms():
    env = make_demo_env(seed=1, n_obstacles=0, food_per_room=12)
    frames = record(env, 100, 1.0, (-4.0, 0.0), None, None)
    save_gif(frames, "rooms")


if __name__ == "__main__":
    food_eating()
    rooms()
