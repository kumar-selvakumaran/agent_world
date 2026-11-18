import json
import subprocess
import sys

import pytest

GUARD = ".claude/hooks/guard_git.py"
DENIED = ["git push --force origin develop", "git push -f origin x", "git push --force-with-lease origin x", "git push origin --delete x", "git push -d origin x",
        "git push --mirror origin", "git push origin +develop", "git push origin :feature/x", "git push -fu origin x", "git -C foo push -f origin x",
        "git status && git push --force origin x", "git branch -d x", "git branch -D x", "git branch --delete x",
        "gh pr merge 1", "gh pr review 1 --approve", "gh pr close 1", "gh pr reopen 1", "gh repo view", "gh api repos/x", "gh issue list", "gh release list",
        "gh secret list", "gh workflow run x", "gh ruleset list", "gh run list", "gh gist list",
        "gh pr create --title t --body 'thanks @someone'"]
ALLOWED = ["git push origin feature/x", "git push -u origin feature/x", "git status", "git branch feature/y", "gh pr create --title t --body 'uses `@types/node`'",
         "gh pr view 1", "uv run pytest -q", "git log --oneline"]


def verdict(command):
    out = subprocess.run([sys.executable, GUARD], input=json.dumps({"tool_input": {"command": command}}), capture_output=True, text=True).stdout
    return "deny" in out


@pytest.mark.parametrize("command", DENIED)
def test_guard_blocks(command):
    assert verdict(command)


@pytest.mark.parametrize("command", ALLOWED)
def test_guard_allows(command):
    assert not verdict(command)
