# agent_world

Read `docs/journal/README.md` first, then `principles.md` and `world-spec.md`.

Standing rules (the full list is in `docs/journal/principles.md`):

- After every effort or decision, append a dated section to `docs/journal/journey.md`.
- After each verified change is merged, tell Kumar so he can try it:
  `uv run python scripts/explore.py --scale 12`.
- One branch per effort off `main`, small atomic commits, every commit passes the tests, merge locally into `main`.
- Use `uv` only. Investigate with throwaway probes in `scratch/<task>.py` that write their output to a text file.
- Raise questions, concerns and blockers as indexed items (Q, C, B), each with a default.
- Docs are append-only. Record a pivot only after it has been agreed.
