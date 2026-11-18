# Physics engine comparison

Which physics engine the world should run on, what was measured, and what was
decided. See the [guide](README.md) for what the labels mean. Written
2026-11-14.

## Decision

- **[Owner]** The engine stays VMAS for now, with a box agent. The puzzles
  described in [world-spec.md](world-spec.md) are possible in it.
- **[Owner]** A switch to MuJoCo is not planned.
- **[Open]** Revisit if a task needs a concave moving body, stable joints or a
  hinged door.

## Candidates

| Engine | Batched on the GPU | Concave bodies | Joints |
| --- | --- | --- | --- |
| VMAS | Yes, native torch | No: shapes are `Box`, `Sphere`, `Line` | Soft; unstable in our probe **[Built]** |
| MuJoCo (MJWarp) | Yes, via Warp, torch interop | Yes, several geoms per body | Hard, standard |
| Box2D | No: CPU, one world per object | Yes | Yes |
| pymunk | No: CPU, one space per world | Yes | Yes |

The Box2D and pymunk rows, licences (VMAS GPL-3.0; MuJoCo Apache-2.0; Box2D and
pymunk MIT) and Python-version notes come from research agents reading
documentation. **[Unverified]** unless stated below.

## Measured

All on one machine: Windows, Python 3.13, a laptop GPU with 4 GB. Other
machines will differ. Probes were run from `scratch/` and are not in the
repository.

**VMAS, whole step, random actions, 256 worlds** (the renderer is ours)

| Entities | Step | Physics only | World-steps/s |
| --- | --- | --- | --- |
| 11 (1 agent) | 79 ms | 18 ms | 3,251 |
| 22 (2 agents) | 216 ms | 31 ms | 1,183 |
| 44 (4 agents) | 781 ms | 75 ms | 328 |
| 80 (6 agents) | 2,008 ms | 170 ms | 128 |

- **[Built]** Physics grows about linearly with entity count. Rendering, which
  scales with pixels x entities x agents, is the cost that grows fastest.
- **[Built]** VMAS physics alone, 11 entities, idle agents: 11,298 / 45,964 /
  177,709 world-steps/s at 256 / 1,024 / 4,096 worlds.

**MuJoCo Warp, a planar world with a U-shaped body of 3 boxes, 13 geoms**

- **[Built]** Physics with 5 substeps per step, driving into walls: 1,517 /
  4,905 / 11,697 world-steps/s at 256 / 1,024 / 4,096 worlds. The walls stopped
  the body. The U body, per-world obstacle layouts and torch interop worked.
- **[Built]** With 1 substep it was about 4x faster but tunnelled through a
  wall, so 1 substep is not usable.
- **[Built]** Its renderer, 64x48 RGB and depth, one camera per world: 34,226 /
  200,067 views/s at 256 / 1,024 worlds with 13 geoms; 16,059 / 31,809 with 77
  geoms. The images were not checked.
- **[Built]** For comparison, our ray caster is about 4,100 views/s (11
  entities) and about 840 views/s (80 entities); both are derived from the
  VMAS runs above, not measured on one identical scene.

**VMAS joints, a U built from three joined boxes**

- **[Built]** The arms lagged by 4 to 40 mm driving straight and spun by more
  than 200 degrees when turning. Stiffer joints gave NaN. More substeps did not
  help. VMAS collisions and joints are soft penalty forces.

## Why VMAS for now

- **[Owner]** The agent is a box, so the missing concave body does not matter.
- **[Proposed]** The puzzles (floor current, rotating patch, thorn field, plank,
  slanted wall, doorway) are tensor logic plus convex pieces. Convex shapes are
  enough for static scenery, because a static wall may be several pieces.
- **[Built]** Physics is not the bottleneck; rendering is. That can be fixed in our
  own ray caster, whichever engine runs the physics.
- **[Proposed]** The scaffolding VMAS gives for free (the environment wrapper,
  `DiffDrive`, random layouts) is already in use.

## Known costs of VMAS

- **[Built]** No concave single bodies; no stable joints.
- **[Built]** Soft contacts: small overlaps, and a fast body can tunnel.
- **[Unverified]** GPL-3.0 licence, and a pyglet pin that needed an override on
  Python 3.13.
- **[Built]** VMAS pins `pyglet<=1.5.27`, which crashes opening a window on
  Python 3.13. `pyproject.toml` overrides it with `pyglet>=2`. VMAS's own
  `render()` is untested with pyglet 2; this project does not use it.
- **[Open]** Whether a bounce off a slanted wall is possible with soft
  contacts.

## VMAS feasibility probes (2026-11-14)

Scratch probes testing whether the planned puzzles can be built in VMAS. Four
were run by research agents and not re-run; the obstacle check was run directly.
The probes are in `scratch/` and not in the repository.

| Element | Result | How |
| --- | --- | --- |
| Rotating patch | **[Built]** Possible, easy | Subclass `DiffDrive`; add the patch rate to the turn command inside the region. A rate above the maximum turn rate cannot be beaten. |
| Floor current | **[Built]** Possible, simple | `env_process_action` hook adds a force; net velocity is command plus current. Diagonal crossing works, a slow straight drive does not. |
| Moving walls | **[Built]** Possible | Set the velocity of a movable landmark each step in `pre_step`. It can tunnel if the agent is pinned against a wall. |
| Chaser and pain on touch | **[Built]** Possible | Scripted actions; `world.is_overlapping` gives a per-world flag. Mask the ghost out. |
| Thorn grid | **[Built]** Possible | A `[B, 10, 10]` timer tensor; pain from the agent's cell. |
| Pushing a plank physically | **[Built]** Not workable | Tunnels or bounces at every substep count tried (1, 2, 5, 10). |
| Kinematic plank | **[Built]** Works | The plank copies the agent's velocity while in contact. Zero pain steps pushing straight. The lag test was not re-run. |
| Bounce off a slanted wall | **[Built]** Possible with a hook | VMAS box contact acts only in a 0.005 shell, so the agent tunnels at speed 0.3 or more. Reflecting heading and velocity in a post-step hook works. |

- **[Built]** In our current world a cylinder obstacle was passed through at
  full speed, and a box obstacle was penetrated and then ejected hard. The
  room's outer walls hold only because of the world-bounds clamp.
- **[Open]** Our own collider for static walls and obstacles, to be decided.

### Correction (later on 2026-11-14)

The plank-pushing and wall-tunnelling failures above were measured with a
**box-shaped agent**. VMAS handles box-against-box contact only in a thin
boundary shell. A circular agent was then tested in the same world (one world
per case, `scratch/sphere_vs_box.py`), driving at speed 1.0:

| Case | Box agent | Circular agent |
| --- | --- | --- |
| Static box obstacle | Penetrated, then ejected hard | Held |
| Static cylinder obstacle | Passed through | Held |
| Plank, length 0.1 or 0.3, pushed | Tunnelled or bounced | Pushed ahead of the agent (plank moved to x = 1.30 and 1.35) |

- **[Built]** With a circular agent VMAS pushes the plank physically and holds
  static obstacles. The prebuilt `transport` scenario also uses circular agents
  to push a box.
- **[Built]** The box agent made contact worse. This weakens the
  kinematic-plank and own-collider proposals above.
- **[Open]** Circular physics with a box drawn inside it, for a visible heading.
  Slanted-wall contact with a circular agent is not yet tested.

### Slanted walls with a circular agent (later on 2026-11-14)

Scratch probes (`scratch/slant_circle*.py`), one world per case, a flat wall
0.1 thick unless stated, agent driving straight at it.

- **[Built]** At speed 0.5 the agent was blocked and slid along the wall at
  90, 60, 45 and 30 degrees.
- **[Built]** At speed 1.0 it was blocked at 90 and 60 degrees, but crossed the
  wall plane at 45 and 30 degrees. A thicker wall (0.2, 0.3) did not fix this.
- **[Built]** Raising VMAS's collision force from 100 to 10,000 held the wall at
  60, 45 and 30 degrees at speed 1.0. A force of 1,000 held 45 but not 30.
- **[Built]** No bounce was seen: the agent blocks and slides. A bounce would
  need a reflection hook.
- **[Open]** Whether a collision force of 10,000 is stable for pushing planks,
  chasers and moving walls. Not tested.

### Collision force and substeps (later on 2026-11-14)

Scratch probes (`scratch/collision_force_*.py`), circular agent, one world per
case, speed 1.0, eight cases: two plank pushes, a moving wall, two agents
head-on, slanted walls at 45 and 30 degrees, a static box and a static cylinder.

- **[Built]** Collision force 100 (default), 1 substep: the plank, moving wall
  and static obstacles held. Two agents closing head-on and the slanted walls
  were crossed.
- **[Built]** Collision force 10,000, 1 substep: unstable. Peak speeds of 50 to
  218 (violent ejections), and the static obstacles and head-on agents were
  crossed. It held the slanted walls, but this is not usable.
- **[Built]** Collision force 1,000 with 5 substeps: peak speeds near 1.0, no
  ejections, everything held except the head-on case and, in one probe, the 30
  degree wall. Collision force 10,000 with 20 substeps also held head-on.
- **[Open]** The 30 degree result depended on the start distance, so it is
  marginal. Substeps multiply the cost of physics.

### Decision: contact settings (later on 2026-11-14)

- **[Owner]** Collision force 1,000 with 5 substeps by default (`WorldConfig`
  `collision_force`, `substeps`).
- **[Built]** Cost on the benchmark (256 worlds, 1 agent, 64x48 view, same GPU):
  2,819 to 1,422 world-steps/s, about 50% slower.
