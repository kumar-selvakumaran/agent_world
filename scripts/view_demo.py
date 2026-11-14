"""Roll out a simple policy and save what agent 0 sees.

    uv run python scripts/view_demo.py

Writes ``outputs/view.gif`` (the rollout) and ``outputs/view.png`` (first frame).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from agent_world import WorldConfig, make_env


def to_image(rgb: torch.Tensor, scale: int) -> Image.Image:
    array = (rgb.clamp(0, 1) * 255).byte().cpu().numpy()
    image = Image.fromarray(array)
    return image.resize((image.width * scale, image.height * scale), Image.NEAREST)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--n-agents", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default=None, help="default: GPU if available")
    args = parser.parse_args()

    env = make_env(
        num_envs=1, device=args.device, seed=args.seed, config=WorldConfig(n_agents=args.n_agents)
    )
    obs = env.reset()
    frames = [to_image(obs[0][0], args.scale)]
    for _ in range(args.steps):
        # Agent 0 drives in a gentle arc; the others turn on the spot.
        actions = [torch.tensor([[0.6, 0.4]], device=env.device)] + [
            torch.tensor([[0.0, 1.0]], device=env.device) for _ in range(args.n_agents - 1)
        ]
        obs, _, _, _ = env.step(actions)
        frames.append(to_image(obs[0][0], args.scale))

    args.out.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.out / "view.png")
    frames[0].save(
        args.out / "view.gif", save_all=True, append_images=frames[1:], duration=100, loop=0
    )
    print(f"Saved {len(frames)} frames to {args.out}/view.gif")


if __name__ == "__main__":
    main()
