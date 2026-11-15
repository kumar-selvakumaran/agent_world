"""The agent lures a chaser into a blue block: the chaser cuts the corner, vanishes and respawns.

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
BLOCK, START, WALK = (-3.0, 0.0), 5, 14
put(env, blocks[0], *BLOCK)
put(env, blocks[1], 6.0, 0.0)
put(env, chasers.landmarks[1], 6.0, -3.0)
put(env, chasers.landmarks[0], -5.4, 0.37)
chasers.heading[:] = 0.0
put(env, agent, -2.0, 1.6, -1.5708)

frames, hit_step = [], None
c = chasers.landmarks[0]
for step in range(300):
    look_from(env, ghost, (-3.7, -1.4), (-3.5, 0.4))
    go = START <= step < START + WALK * 1
    cmd = [0.9 if go else 0.0, 0.0]
    prev = c.state.pos[0].clone()
    obs, *_ = env.step([torch.tensor([cmd], device=env.device), torch.zeros(1, 2, device=env.device)])
    if hit_step is None and (c.state.pos[0] - prev).norm() > 1:
        hit_step = step
        print("chaser vanished from the block at step", step)
    if step % 3 == 0:
        frames.append(to_image(obs[-1][0], 2))
    if hit_step and step > hit_step + 45:
        break
save_gif(frames, "blue_block_lure", 0.12)
