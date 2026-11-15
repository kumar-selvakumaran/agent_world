# Physics engine comparison

Which physics engine the world should run on, what was measured, and what was
decided. See the [guide](README.md) for what the labels mean. Written
2026-11-14.

## Decision

- **[Kumar]** The engine stays VMAS for now, with a box agent. The puzzles
  described in [world-spec.md](world-spec.md) are possible in it.
- **[Kumar]** A switch to MuJoCo is not planned.
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

All on one machine: Windows, Python 3.13, RTX 3050 Laptop GPU with 4 GB. Other
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

- **[Kumar]** The agent is a box, so the missing concave body does not matter.
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
- **[Open]** Whether a bounce off a slanted wall is possible with soft
  contacts.
