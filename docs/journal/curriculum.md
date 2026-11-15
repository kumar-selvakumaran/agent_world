# Learning curriculums

Ways to condition, train or otherwise enable learning in the agent. See the
[guide](README.md) for what the labels mean. Started 2026-11-14.

## Why this exists

- **[Kumar]** The north-star scenario (the mirror puzzle in
  [world-spec.md](world-spec.md)) needs concepts the agent has no basis for yet:
  light, line of sight, reflection.
- **[Kumar]** The question is how to expose a curriculum to the agent so that it can
  learn them, and which kinds of signal can be used to train it.
- **[Open]** No agent or learning method exists yet. This document records strategies,
  not a plan.

## Strategies

### 1. Demonstrations by scripted agents

- **[Kumar]** Pre-programmed agents with hard-coded, deterministic policies solve
  problems in front of the learning agent, for example unblocking things, so that the
  learner comes to understand how the world works.
- **[Built]** The building block exists: the chasers are scripted agents with
  deterministic policies (`chaser.py`), added to the world as landmarks, not agents.
- **[Proposed]** What it would need:
  - The demonstrator's actions and their effects must be visible in the learner's
    view, for example a lever moving, a pillar turning and a beam shifting.
  - Demonstrations that are reproducible, because the policies are deterministic, and
    varied in their start states.
- **[Proposed]** Risks:
  - Watching shows what happens, not necessarily why.
  - Small effects are hard to see at 64x48 (the legibility concern).
- **[Open]** What the learner is trained on while it watches. Not defined.

### More strategies

- **[Open]** To be added as Kumar describes them. Kumar has said he has ideas for
  other kinds of signal.
