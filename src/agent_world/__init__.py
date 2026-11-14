"""Agent World: a 2.5D third-person world built on VMAS."""

from agent_world.camera import CameraConfig
from agent_world.raycast import Palette
from agent_world.scenario import Scenario, WorldConfig

__all__ = ["CameraConfig", "Palette", "Scenario", "WorldConfig", "make_env"]


def make_env(num_envs: int = 1, device: str = "cpu", config: WorldConfig | None = None, **kwargs):
    """Create the VMAS environment for this world.

    ``kwargs`` are forwarded to :func:`vmas.make_env` (e.g. ``max_steps``, ``seed``).
    """
    import vmas

    return vmas.make_env(
        scenario=Scenario(),
        num_envs=num_envs,
        device=device,
        config=config or WorldConfig(),
        **kwargs,
    )
