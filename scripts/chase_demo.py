"""Record a chase as the ghost spectator sees it.

    uv run python scripts/chase_demo.py

The agent stands still until a chaser bumps it, turns a full circle to see what hit it,
turns away and sprints. The spectator stays close beside the chase and the chasers
ignore it. Writes ``outputs/chase.gif``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from PIL import Image

from agent_world import CameraConfig, WorldConfig, make_env


def to_image(rgb: torch.Tensor, scale: int) -> Image.Image:
    array = (rgb.clamp(0, 1) * 255).byte().cpu().numpy()
    image = Image.fromarray(array)
    return image.resize((image.width * scale, image.height * scale), Image.NEAREST)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=170)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default=None, help="default: GPU if available")
    args = parser.parse_args()

    config = WorldConfig(
        n_agents=1, spectator=True, n_obstacles=0, food_per_room=0,
        camera=CameraConfig(width=192, height=144),
    )
    env = make_env(num_envs=1, device=args.device, seed=args.seed, config=config)
    env.reset()
    agent, ghost = env.agents
    chasers = env.scenario._chasers
    dev = env.device

    def put(entity, x, y, heading=0.0):
        entity.set_pos(torch.tensor([[x, y]], device=dev), batch_index=None)
        entity.set_rot(torch.full((1, 1), heading, device=dev), batch_index=None)

    put(agent, -5.6, -3.2)
    for i, (x, y, heading) in enumerate([(-7.4, -3.2, 0.0), (-7.4, 3.4, math.pi / 2)]):
        put(chasers.landmarks[i], x, y, heading)
        chasers.heading[:, i] = heading
        chasers.landmarks[i].set_vel(torch.zeros(1, 2, device=dev), batch_index=None)
    print("Chaser 0 starts 1.8 units behind the agent; chaser 1 looks away.")

    frames, bumps, first_bump = [], 0, None
    for step in range(args.steps):
        t = step * 0.1
        # The spectator stands to the left of the chase, 1.8 units away, looking at its middle.
        a_xy, c_xy = agent.state.pos[0], chasers.landmarks[0].state.pos[0]
        mid, ahead = (a_xy + c_xy) / 2, a_xy - c_xy
        ahead = ahead / ahead.norm().clamp(min=1e-6)
        left = torch.stack([-ahead[1], ahead[0]])
        put(ghost, *(mid + 1.8 * left).tolist(), math.atan2(-left[1].item(), -left[0].item()))
        since = None if first_bump is None else t - first_bump
        if since is None:  # stand still until the chaser walks up and bumps it
            command = [0.0, 0.0]
        elif since < 4.5:  # one full turn on the spot (about 1.4 rad/s), to see what hit it
            command = [0.0, 1.0]
        elif since < 5.7:  # turn north, away from the chaser
            command = [0.6, 1.0]
        else:  # sprint, faster than the chaser
            command = [1.0, 0.0]
        actions = [torch.tensor([command], device=dev), torch.zeros(1, 2, device=dev)]
        obs, _, _, _ = env.step(actions)
        touched = bool(chasers.touched.any())
        bumps += touched
        if touched and first_bump is None:
            first_bump = t
        if step % 2 == 0:
            frames.append(to_image(obs[-1][0], args.scale))  # the spectator's view

    args.out.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.out / "chase.gif", save_all=True, append_images=frames[1:], duration=200, loop=0)
    print(f"Saved {len(frames)} frames to {args.out}/chase.gif ({bumps} bump steps)")


if __name__ == "__main__":
    main()
