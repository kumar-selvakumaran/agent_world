"""Walk around the world yourself as the spectator (a ghost agent).

    uv run python scripts/explore.py

Controls: W/S or Up/Down move, A/D or Left/Right turn, R resets, Esc quits.
The spectator sees what any agent would see, but passes through everything
and is invisible to the other agents, which stand still here.
"""

from __future__ import annotations

import argparse

import pyglet
import torch
from pyglet.window import key

from agent_world import CameraConfig, WorldConfig, make_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-agents", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=240, help="view width in pixels")
    parser.add_argument("--height", type=int, default=160, help="view height in pixels")
    parser.add_argument("--scale", type=int, default=4, help="window = view * scale")
    parser.add_argument("--device", default=None, help="default: GPU if available")
    parser.add_argument("--max-frames", type=int, default=None, help="quit after this many frames")
    args = parser.parse_args()

    config = WorldConfig(
        n_agents=args.n_agents,
        spectator=True,
        camera=CameraConfig(width=args.width, height=args.height),
    )
    env = make_env(num_envs=1, device=args.device, seed=args.seed, config=config)
    state = {"obs": env.reset(), "frames": 0}

    window = pyglet.window.Window(
        args.width * args.scale, args.height * args.scale, caption="Agent World - spectator"
    )
    keys = key.KeyStateHandler()
    window.push_handlers(keys)

    def step(_dt: float) -> None:
        if keys[key.R]:
            state["obs"] = env.reset()
            return
        forward = (keys[key.W] or keys[key.UP]) - (keys[key.S] or keys[key.DOWN])
        turn = (keys[key.A] or keys[key.LEFT]) - (keys[key.D] or keys[key.RIGHT])
        actions = [torch.zeros(1, 2, device=env.device) for _ in range(args.n_agents)]
        # The spectator is the last agent.
        actions.append(torch.tensor([[float(forward), float(turn)]], device=env.device))
        state["obs"], _, _, _ = env.step(actions)
        state["frames"] += 1
        if args.max_frames is not None and state["frames"] >= args.max_frames:
            pyglet.app.exit()

    @window.event
    def on_draw() -> None:
        rgb = (state["obs"][-1][0].clamp(0, 1) * 255).byte().cpu().numpy()
        image = pyglet.image.ImageData(
            args.width, args.height, "RGB", rgb.tobytes(), pitch=-args.width * 3
        )
        window.clear()
        image.blit(0, 0, width=window.width, height=window.height)

    pyglet.clock.schedule_interval(step, 1 / 20)
    pyglet.app.run()


if __name__ == "__main__":
    main()
