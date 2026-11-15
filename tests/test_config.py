import os

from agent_world import config


def test_downloads_and_outputs_live_inside_the_project():
    for folder in (config.DATA_DIR, config.ASSETS_DIR, config.OUTPUTS_DIR):
        assert folder.parent == config.PROJECT_DIR
    assert (config.PROJECT_DIR / "pyproject.toml").exists()
    assert os.environ["HF_HOME"] and os.environ["TORCH_HOME"]
