# agent_world

Read `docs/journal/README.md` first, then `principles.md` and `world-spec.md`.

Standing rules (the full list is in `docs/journal/principles.md`):

- After every effort or decision, append a dated section to `docs/journal/journey.md`.
- After each verified change is merged, tell Kumar so he can try it:
  `uv run python scripts/explore.py --scale 12`.
- Branches: one per effort, named feature/<name>, fix/<name> or docs/<name>, off `develop` (or off the branch of an
  open pull request it depends on: stacked PRs). Small, non-additive commits may go straight to `develop`.
- Commits: small and atomic; every commit passes `uv run pytest`; title, a body that says why, attribution line.
- Pull requests: push the branch and open a PR into `develop` (or the branch it stacks on) with `gh pr create`;
  description: what, why, how verified, what is not covered. Only Kumar reviews and merges. After a lower PR merges,
  retarget the PR stacked on it to `develop`.
- Pushing: safe, verified, secret-free code may be pushed. Journals and docs need Kumar's confirmation before they
  enter git. Never force-push. Never delete branches.
- Code reaches GitHub from the code-only clone `../agent_world_worktrees/code-only` (`develop`, no docs/journal); after
  a PR merges, cherry-pick it into the local working repository, which also holds the journals.
- `main` on GitHub is the default branch and stays blank until reviewed content is merged into it through a PR.
- Costly, risky or critical-environment commands follow the hitl-dev skill; step files live in scratch/hitl/.
- Use `uv` only. Investigate with throwaway probes in `scratch/<task>.py` that write their output to a text file.
- Ask questions with the question wizard (the AskUserQuestion tool), the default as the first option marked Recommended. Raise concerns and blockers as indexed text (C, B), each with a default.
- Docs are append-only. Record a pivot only after it has been agreed.
