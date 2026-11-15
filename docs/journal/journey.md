# Journey

The path from a research question about reasoning to building this world.
Chronological; each section says what was explored and what it left behind.
See the [guide](README.md) for the labels and for how far to trust the paper
references (most are unverified).

## The question

**[Kumar]** How can a system do human-like System 2 reasoning: slow,
deliberate, built up step by step, and checked against what it understands of
the world? The session began with GFlowNets as a candidate mechanism and ended
with building an environment to test such systems.

## 1. GFlowNets as System 2 samplers

- Bengio frames System 2 as structured, stepwise reasoning. GFlowNets build an
  object step by step and sample in proportion to a reward instead of
  maximising it, so they keep several hypotheses alive.
- Reference point: Hu et al. (2023), which fine-tunes a language model as a
  GFlowNet to sample chains of thought, treating the chain as a latent variable
  between question and answer.
- Kumar's alternatives were examined: asking the model to rationalise a given
  answer, and iterating guess-reason-revise. Both are known ideas with known
  weaknesses; they complement GFlowNet training but do not replace it.

## 2. What reasoning is

- **[Kumar]** People do not search end to end. They posit intermediate
  subgoals, check each for local consistency with their world model, and build
  up.
- **[Kumar]** Reasoning is **consistency with one's own subjective, imperfect
  world model**, not provable correctness. People reason with an imprecise
  model *and* an imprecise checker, and correct course.
- **[Agreed]** Two conditions keep this from being empty: the steps must do
  real work (changing a step changes the conclusion), and the reasoner must
  revise when it meets a contradiction.
- **[Kumar] Core thesis.** Reasoning is sampling or optimising consistency
  *over the space of logically constructible trajectories*: logic defines the
  search space, an energy ranks within it.
- **[Kumar]** Neural networks are unsound *and* incomplete approximations of
  the valid set, and this is *more* human-like than a sound, complete search.
- **[Kumar]** Changing a premise reshapes the energy landscape itself. This
  goes beyond LeCun's planner, where the world model and cost stay fixed.
- **[Agreed]** If an action can reach any latent state, the search space is a
  formality. Resolution: keep states continuous but **moves discrete**.
- Kambhampati's position was discussed. (corrected) His demand for provable
  correctness concerns *trusting* output, not what counts as reasoning.

## 3. A first agent design

- **[Proposed]** A full specification: a JEPA world model (encoder plus
  action-conditioned predictor); a fast System 1 (policy, subgoal proposer,
  reachability estimate); and a System 2 GFlowNet over partial plans whose
  actions insert subgoals or expand segments, trained with sub-trajectory
  balance, with good plans distilled back into System 1.
- **[Kumar] Blocker.** There is no data or reward that labels logical
  constructibility.
- **[Proposed]** Ways round it: put constructibility in the action space; use
  domains with exact checkers; use world-model prediction error as a soft
  label. Kumar noted that the first re-introduces a hard-coded bias. Middle
  ground: hard-code only minimal primitive moves and learn the rest.
- **[Agreed]** A GFlowNet is neutral. Inductive biases enter through the
  states and actions (what can exist), the reward (what is preferred) and the
  policy architecture (how it generalises).
- Compared with LeCun's H-JEPA: both reason in latent space with subgoals and
  a learned world model. The proposal differs in sampling rather than
  per-query optimisation, discrete moves, decomposition as an action, and
  revising premises.

## 4. Goyal & Bengio's inductive biases

- Ten biases were extracted from the paper: two systems; verbalisable
  high-level variables; semantic variables are causal; independent mechanisms;
  localised changes; stable versus unstable knowledge; a sparse factor graph;
  reusable schemas; short causal chains; top-down and bottom-up integration.
- (corrected) The paper does **not** propose GFlowNets for sampling thoughts.
  It is conceptual, with almost no formal definitions.
- **[Kumar]** Concepts are too many to list, people do not hold explicit
  causal graphs, and verbal accounts of our reasoning are lossy and often
  wrong. So variables and factors should not be explicit or predefined.
- **[Kumar]** Slots and factors must never be tied to predefined notions, and
  **being verbalisable must not be a requirement**; concepts should be free to
  be abstract.
- **[Agreed]** Knowledge stays implicit. Only a small **workspace** is
  explicit: a temporary structure built, checked and repaired during
  reasoning. A reasoning step is an edit to it.
- **[Agreed]** The problem splits into four parts: the **space** reasoning
  happens in, the **structure** imposed on it, the **operations** allowed, and
  the **coherence** measure. Keeping coherence separate lets different
  structures be compared under one measure. Written up in LaTeX for Kumar's
  notes.

## 5. Reading the paper closely

- On Kumar's challenge, Claude re-read the paper and grounded the list in it.
  Operations it describes: select by attention, bind, match and instantiate
  schemas, operate on few variables, seek coherence, broadcast, write to
  working memory, intervene, compose, reason over categories, recall,
  consolidate. Analogy and retracting a premise are **not** in the paper.
- **[Kumar]** Types and matching need not be explicit; they can be soft. The
  paper agrees ("distributed types").
- What System 2 adds over System 1, per the paper: flexible queries, a
  sequential and variable amount of computation, recombination on the fly, and
  immediate use of new knowledge.
- **[Kumar]** The paper's random variables need a defined support. A concept
  need not be a block of coordinates; it can be a latent factor spread across
  the whole activation.

## 6. Reasoning inside a language model

- **[Kumar]** Language models hold a useful world model, though inaccurate,
  incomplete and possibly inconsistent. The architecture fixes a class of
  learnable world models; training picks a useful one. The goal is to reason
  over the concepts in *that* model, the way people do.
- **[Kumar] Faithfulness concern.** Any map from the model's representation to
  one a reasoner can manipulate may *lose* what the model knows, or *add*
  knowledge of its own, so that the reasoning no longer reflects the model.
- **[Kumar]** So the reasoner should be minimal and integrated: an
  **alternative pathway inside the model**, which the model can choose instead
  of its native route. Stated formally as finding a model in the System 2
  class that is faithful to the one the language model learned.
- Related work found (mostly from search): Latent Thought Flow (abstract only),
  the coprocessor line, Coconut, adaptive and latent reasoning routers. None
  structures the latent space with these priors or tests faithfulness.
- **[Kumar]** A cost on computation could motivate the model to take a more
  efficient abstract route. **[Agreed]** as one term beside accuracy.
- **[Proposed]** A minimal experiment: k-hop function composition, training on
  few hops and testing on more, with baselines matched for size and compute.
- **[Kumar]** After asking whether human reasoning is known to follow the
  priors (it is not established), Kumar **lost confidence in all of them
  except sparsity and degree of abstraction**, and set factor graphs aside.

## 7. LeCun and Bengio side by side

- Compared from the three papers read directly. LeCun: reasoning as energy
  minimisation through a world model; abstraction from predictability over
  time; a hierarchy with no capacity limit. Bengio: reasoning as selecting and
  combining a few variables in a small workspace; abstraction from meaning and
  causal role.
- **[Kumar]** Agrees with Bengio that we know far more than we are thinking
  about at any moment: thought works on a sparse selection.
- **[Kumar]** We cannot assume knowledge is *stored* as a sparse factor graph;
  storage is messy. The structure belongs to the workspace, the form thoughts
  take when in focus.
- **[Kumar]** Reasoning works on *projections* of raw observations onto the
  world model. **[Agreed]** No paper treats that projection as part of
  reasoning, revisable on contradiction.
- Global workspace theory, and a reported result of a verbalisable workspace
  inside language models, were reviewed as background (unverified).
- Why JEPA world models are rare in games: collapse, including dropping
  whatever is hard to predict; no rewards; determinism; slow planning.

## 8. The turn to building a world

- **[Kumar]** Build an environment that *demands* the properties of a System 2
  reasoner; define its requirements precisely; then build general agents for
  it. The requirements are in [world-spec.md](world-spec.md).
- Candidates reviewed: Kinetix, VMAS, particle environments, XLand-MiniGrid,
  PHYRE, Craftax, MiniHack. **[Kumar]** chose a multi-agent world with basic
  physics, on VMAS.
- **[Kumar]** Fixed rules can still give complex behaviour, as in the Game of
  Life. **[Agreed]**, with the aim of behaviour that compresses well at a
  higher level.
- **[Kumar]** How to give an agent something like pain, and a bias towards
  survival. Discussed: valence, an immediate intrinsic cost, homeostasis,
  death as an absorbing state, and bodily signals as an input alongside vision.
- **[Kumar]** The scenario: a safe room with scarce food, doors to riskier and
  richer areas, and a door the agent must first learn to open.
- **[Kumar]** The view: a raised eye looking down, third person, with heights
  for vision over 2D physics.

## 9. First build (2026-11-14)

- **[Built]** The repository was set up as a uv project with VMAS as a
  dependency, and the world as a VMAS custom scenario: a walled room,
  obstacles of varied heights, steerable agents, and the third-person
  perspective view rendered by ray casting.
- **[Built]** A spectator for a person to explore the world as a ghost.
- **[Built]** Automatic use of a GPU when present, a benchmark script, and
  instructions for running it.
- Found on the way: rendering, not physics, is the cost. A first tuning pass
  made it about four times faster on CPU.
- Standing instructions given this day are recorded in
  [principles.md](principles.md).

## 10. First GPU run (2026-11-14)

- Found that the environment held a CPU-only build of PyTorch, so the GPU had
  never been used. Pointed torch at the CUDA index (cu128) in `pyproject.toml`.
- **[Built]** Measured on an RTX 3050 Laptop GPU (4 GB), 256 worlds, 1 agent,
  64x48 view: about 3,480 world-steps per second on the GPU, against about 478
  on the CPU of the same machine. This partly settles the GPU thread below; it
  says nothing yet about whether that is enough for large-scale training.
- **[Open]** The 4 GB of GPU memory bounds the number of parallel worlds.
  Chunking the rendering is one way round it; not yet investigated.

## 11. Which engine, and what the world is for (2026-11-14)

- **[Kumar]** Wanted flexible geometry, such as a U-shaped gripper body.
  VMAS has only box, sphere and line shapes, so a concave body was tested with
  joints. **[Built]** The joined U was unstable; see
  [physics-engines.md](physics-engines.md).
- Four engines were compared: VMAS, MuJoCo (MJWarp), Box2D and pymunk. Only
  VMAS and MuJoCo batch worlds on the GPU. Measurements are in the comparison.
- **[Built]** The MuJoCo renderer is much faster than our ray caster, but its
  physics was 8 to 15 times slower than VMAS's, and one substep tunnelled.
- **[Kumar]** The puzzles to build are tensor logic on top of convex shapes: a
  floor current, a rotating patch before a doorway, a thorn field cleared by a
  pushed plank, and a slanted wall. A door was dropped. See
  [world-spec.md](world-spec.md).
- **[Kumar]** The gripper is set aside: the agent is a box, and the engine stays
  VMAS. **[Built]** The agent is now a box. A switch to MuJoCo is not planned.
- **[Open]** Whether contacts with a slanted wall can bounce, since VMAS
  contacts are soft penalty forces.

## 12. Building the world up (2026-11-14, later)

- **[Built]** The VMAS probes showed that the earlier failures (walls and
  obstacles not holding, a plank not pushable) came from a box-shaped agent.
  The agent is a circle in the physics, drawn as a box inside it.
- **[Kumar]** Contacts: collision force 1,000 with 5 substeps. **[Built]** It
  roughly halves throughput. See [physics-engines.md](physics-engines.md).
- **[Kumar]** Two 8 by 8 rooms joined by a doorway. **[Built]**
- **[Kumar]** A rotating pink patch in the doorway, stronger than the agent can
  turn back (2.5 rad/s), with arrowheads twice as big as first built.
  **[Built]** It can be crossed by turning before the patch and slowing the spin.
- **[Kumar]** Second scenario: chasers (0.7 of the agent's speed, they see only
  within range, angle and line of sight, stop when they lose sight, and bump an
  agent they touch), consumable food drawn on the floor, blue blocks that make
  a touching chaser respawn far away, and red-and-black striped chasers.
  **[Open]** Being built in separate git worktrees.
- **[Kumar]** The explorer works with the new rooms and patch. After each
  verified change Kumar wants to be told, to try it in the explorer.
- **[Kumar]** Journey and rules: this file is updated after every effort.

## 13. Chasers and food (2026-11-14, later)

- **[Built]** Scripted chasers: two per room, red and black striped, 0.7 of the
  agent's speed. They see only within 5 units, 60 degrees and a clear line of
  sight, stop when they lose sight, are spun by the patch, and bump an agent
  they touch by 0.5 units over 3 steps with a 1 s cooldown.
- **[Built]** Food: 24 ochre discs per room painted on the floor, eaten by
  touching, respawning after 5 s. About 6.6 discs are in view on average.
- **[Built]** The patch arrowheads are twice as big, inside the disc.
  **[Kumar]** The patch turn rate of 2.5 rad/s feels right.
- **[Built]** Project paths in `config.py`, hard-linked environments, and a
  `.dockerignore`.
- **[Built]** Found and fixed a crash on reset with food switched off.
- Chasers and food were built in parallel in separate git worktrees, then merged
  one at a time. Merging them exposed a flaky test: chasers bumped the agent in
  contact tests, so those tests now switch chasers off.
- **[Built]** Throughput at 256 worlds is about 300 to 400 world-steps/s. The
  chasers cost the most (more than half), then the 5 substeps (about a half).
  Food and the patch cost almost nothing. Measured while the explorer may have
  been running.
- **[Kumar]** The scenario code is to be split into a package (layout B: VMAS
  adapters in `scenario/`, pure behaviours outside). **[Open]** Not yet done.
- **[Open]** Blue blocks that make a touching chaser respawn far away; making
  the chasers cheaper.

## Open threads

- **[Open]** Picture or entity list as the observation.
- **[Open]** GPU performance, and whether it is enough for large-scale training.
- **[Open]** Which of the remaining priors, beyond sparsity and abstraction,
  the world should test for.
- **[Open]** How the world's tasks and rewards will make hierarchy, memory and
  learned rules necessary rather than merely possible.
- **[Open]** What the first agent is: the language-model pathway of section 6,
  the JEPA-plus-GFlowNet design of section 3, or something simpler.
- **[Open]** How faithfulness to an existing world model would be measured.
