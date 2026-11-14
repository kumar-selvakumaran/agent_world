"""Measure simulation speed: world-steps per second, with random actions.

    uv run python scripts/benchmark.py --num-envs 1024
"""

from __future__ import annotations

import argparse
import time

import torch

from agent_world import CameraConfig, WorldConfig, make_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--n-agents", type=int, default=1)
    parser.add_argument("--width", type=int, default=CameraConfig.width)
    parser.add_argument("--height", type=int, default=CameraConfig.height)
    parser.add_argument("--device", default=None, help="default: GPU if available")
    args = parser.parse_args()

    config = WorldConfig(
        n_agents=args.n_agents, camera=CameraConfig(width=args.width, height=args.height)
    )
    env = make_env(num_envs=args.num_envs, device=args.device, config=config)
    env.reset()

    def step() -> None:
        env.step(
            [torch.rand(args.num_envs, 2, device=env.device) * 2 - 1 for _ in env.agents]
        )

    def wait() -> None:  # GPU work is asynchronous; wait for it before timing
        if torch.device(env.device).type == "cuda":
            torch.cuda.synchronize()

    step()  # warm-up
    wait()
    start = time.perf_counter()
    for _ in range(args.steps):
        step()
    wait()
    elapsed = time.perf_counter() - start

    rate = args.num_envs * args.steps / elapsed
    print(
        f"{rate:,.0f} world-steps/s on {env.device} "
        f"({args.num_envs} worlds, {args.n_agents} agent(s), {args.width}x{args.height} view)"
    )


if __name__ == "__main__":
    main()
