# agent_world

A small 2.5D world for studying agents that have to reason, built as a custom
scenario on top of [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator).

The physics is flat (VMAS, 2D, batched over many worlds). Vision is not: every
entity is extruded to a height and seen through a third-person perspective
camera, so agents get perspective, occlusion and their own body in view.

## Setup

Everything runs in a project-local environment managed by [uv](https://docs.astral.sh/uv/):

```bash
uv sync                               # creates .venv and installs everything
uv run pytest                         # tests
uv run python scripts/view_demo.py    # saves outputs/view.gif (what agent 0 sees)
uv run python scripts/explore.py      # walk around yourself (needs a display)
uv run python scripts/benchmark.py    # world-steps per second on this machine
```

### PyTorch and the GPU

`pyproject.toml` takes `torch` from PyTorch's CUDA 12.8 index
(`https://download.pytorch.org/whl/cu128`), because the default PyPI wheel on
Windows is CPU-only. This suits an NVIDIA GPU on Windows or Linux. On a machine
without one it still runs, on the CPU; on a platform that index does not cover
(for example macOS), remove the `[tool.uv.sources]` and `[[tool.uv.index]]`
entries and run `uv sync` again. Check what you got with:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### The explorer window and pyglet

VMAS pins `pyglet<=1.5.27`, which crashes when opening a window on Python 3.13
(`NameError: name 'COMPointer' is not defined`). `pyproject.toml` therefore
overrides it with `pyglet>=2` (`[tool.uv] override-dependencies`). This project
draws its own views and uses pyglet only for the explorer window. VMAS's own
`render()` has not been tried with pyglet 2.

### Credits

This project depends on [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator),
which is licensed under GPL-3.0.

## The world

Two walled 8 by 8 rooms joined by a doorway, with a few static obstacles (boxes and cylinders of
different heights) and one or more agents. Each agent is a circle in the physics and is drawn as a small box inside it (`agent_size`), so its heading is visible. Agents steer like a vehicle: the
action is `(forward speed, turn rate)`, each in `[-1, 1]`. There is no task
yet; the reward is zero.

```python
import torch
from agent_world import WorldConfig, make_env

env = make_env(num_envs=32, config=WorldConfig(n_agents=2))
obs = env.reset()                     # one tensor per agent: [32, 48, 64, 3]
actions = [torch.zeros(32, 2, device=env.device) for _ in env.agents]
obs, rewards, dones, info = env.step(actions)
```

`make_env` puts everything (physics and rendering) on the GPU when there is
one, and on the CPU otherwise; pass `device="cpu"` or `device="cuda:1"` to
choose. Observations come back on `env.device`, so create actions there too.

### The view

The eye sits `eye_height` above the ground, above the agent and taller than
it, looking along the agent's heading and tilted down. The view is bounded by

- the **horizon** at the top,
- the ground a little **behind the agent** at the bottom (`rear_distance`),
- **60 degrees** to each side (`half_fov_deg`).

It is an ordinary pinhole camera, so straight lines stay straight. See
`CameraConfig` for the knobs, including `eye_back` to pull the eye back behind
the agent.

### The spectator

`WorldConfig(spectator=True)` adds one extra agent for a human. It moves and
sees exactly like the others, but it is a ghost: it passes through agents,
obstacles and walls, cannot push anything, and nobody else can see it. Only
the room's outer bounds stop it. It is always the last agent in `env.agents`.

Leave it off for training: each agent costs one rendered view per step.

## Layout

```
src/agent_world/
  camera.py     pose -> viewing rays            (pure tensor maths)
  raycast.py    rays + scene -> colour, depth   (pure tensor maths)
  scenario.py   the VMAS scenario: entities, physics, observation
scripts/        view_demo.py, explore.py
tests/
```

`camera.py` and `raycast.py` know nothing about VMAS, so the view can be
changed or replaced without touching the world, and the other way round.

## Performance

Rendering dominates: physics is about 1 ms per batched step, the view is
`pixels x entities` ray tests per world. On 2 CPU threads, a 64x48 view with
11 entities runs at roughly 750 world-steps per second. It is all batched
tensor code, so it should run much faster on a GPU; measure it on your machine
with `scripts/benchmark.py`.
