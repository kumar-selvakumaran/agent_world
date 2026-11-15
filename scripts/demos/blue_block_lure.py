"""The agent lures a chaser into a blue block: it hides just behind the block's corner, the chaser clips the block and vanishes.

    uv run python scripts/demos/blue_block_lure.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import look_from, make_demo_env, put, save_gif, to_image

env = make_demo_env(chasers_per_room=1, blocks_per_room=1)
agent, ghost = env.agents
chasers, blocks = env.scenario._chasers, env.scenario._blocks
c = chasers.landmarks[0]
put(env, blocks[0], -4.0, 0.0)
put(env, blocks[1], 6.0, 0.0)
put(env, chasers.landmarks[1], 6.0, -3.0)
put(env, c, -6.6, 0.0)
chasers.heading[:] = 0.0
put(env, agent, -1.6, -0.55, 3.14159)

WALK = 35
frames, hit_step = [], None
for step in range(220):
    a, b = agent.state.pos[0], c.state.pos[0]
    mid = (a + b + torch.tensor([-4.0, 0.0], device=a.device)) / 3
    look_from(env, ghost, (mid[0].item(), -2.0), (mid[0].item(), mid[1].item() + 0.2))
    cmd = [0.4 if step < WALK else 0.0, 0.0]
    prev = b.clone()
    obs, *_ = env.step([torch.tensor([cmd], device=env.device), torch.zeros(1, 2, device=env.device)])
    if hit_step is None and (c.state.pos[0] - prev).norm() > 1:
        hit_step = step
        print("chaser vanished from the block at step", step)
    if step % 3 == 0:
        frames.append(to_image(obs[-1][0], 2))
    if hit_step and step > hit_step + 30:
        break
print("agent", agent.state.pos[0].tolist())
save_gif(frames, "blue_block_lure", 0.12)
