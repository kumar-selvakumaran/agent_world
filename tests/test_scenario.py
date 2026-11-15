import pytest
import torch

from agent_world import CameraConfig, Scenario, WorldConfig, make_env
from agent_world.scenario import AGENT_COLORS, SPECTATOR_COLOR

CAMERA = CameraConfig(width=48, height=32)


def below_eye_pixel(env):
    """Index (row, col) of the pixel looking most nearly straight down."""
    dirs = env.scenario._local_dirs.reshape(CAMERA.height, CAMERA.width, 3)
    flat = dirs[..., 2].argmin()
    return divmod(flat.item(), CAMERA.width)


def test_observation_shape_and_range():
    env = make_env(num_envs=3, seed=0, config=WorldConfig(n_agents=2, camera=CAMERA))
    obs = env.reset()
    assert len(obs) == 2
    for o in obs:
        assert o.shape == (3, CAMERA.height, CAMERA.width, 3)
        assert o.min() >= 0 and o.max() <= 1
    assert env.observation_space[0].shape == (CAMERA.height, CAMERA.width, 3)


def test_agent_sees_itself_below_the_eye():
    env = make_env(num_envs=4, seed=0, config=WorldConfig(n_agents=2, camera=CAMERA))
    obs = env.reset()
    row, col = below_eye_pixel(env)
    for i in range(2):
        expected = torch.tensor(AGENT_COLORS[i], device=env.device).expand(4, 3)
        assert torch.allclose(obs[i][:, row, col], expected, atol=1e-5)


def test_stepping_moves_agent_and_changes_view():
    env = make_env(num_envs=2, seed=0, config=WorldConfig(camera=CAMERA))
    first = env.reset()[0]
    agent = env.agents[0]
    start = agent.state.pos.clone()
    for _ in range(10):
        obs, rewards, _, _ = env.step([torch.tensor([[1.0, 0.5]]).expand(2, 2)])
    assert (agent.state.pos - start).norm(dim=-1).min() > 0.05
    assert not torch.allclose(obs[0], first)
    assert (rewards[0] == 0).all()


def test_agents_stay_inside_the_room():
    cfg = WorldConfig(n_agents=3, spectator=True, camera=CAMERA)
    env = make_env(num_envs=8, seed=1, config=cfg)
    env.reset()
    for _ in range(150):
        env.step([torch.tensor([[1.0, 0.0]]).expand(8, 2) for _ in env.agents])
    limit = cfg.arena_size / 2 - cfg.wall_thickness / 2 - cfg.agent_radius
    for agent in env.agents:
        assert agent.state.pos.abs().max() <= limit + 1e-2


def test_spectator_is_a_ghost():
    cfg = WorldConfig(n_agents=1, n_obstacles=0, spectator=True, camera=CAMERA)
    env = make_env(num_envs=1, seed=0, config=cfg)
    env.reset()
    agent, spectator = env.agents
    assert not spectator.collide

    # Overlap the spectator with the agent, directly in front of it.
    heading = agent.state.rot
    ahead = torch.cat((torch.cos(heading), torch.sin(heading)), dim=-1)
    spectator.set_pos(agent.state.pos + 0.15 * ahead, batch_index=None)
    spectator.set_rot(heading.clone(), batch_index=None)
    before = agent.state.pos.clone()
    obs, _, _, _ = env.step([torch.zeros(1, 2, device=env.device)] * 2)
    assert torch.allclose(agent.state.pos, before, atol=1e-6)  # not pushed away

    # The agent does not see the spectator; the spectator sees itself.
    row, col = below_eye_pixel(env)
    assert torch.allclose(obs[0][0, row, col], torch.tensor(AGENT_COLORS[0], device=env.device), atol=1e-5)
    assert torch.allclose(obs[1][0, row, col], torch.tensor(SPECTATOR_COLOR, device=env.device), atol=1e-5)
    assert not (obs[0] == torch.tensor(SPECTATOR_COLOR, device=env.device)).all(-1).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_runs_on_gpu():
    cfg = WorldConfig(n_agents=2, spectator=True, camera=CAMERA)
    env = make_env(num_envs=4, config=cfg)  # no device given: picks the GPU
    obs = env.reset()
    obs, rewards, _, _ = env.step([torch.zeros(4, 2, device=env.device) for _ in env.agents])
    assert all(o.device.type == "cuda" for o in obs + rewards)
    row, col = below_eye_pixel(env)
    assert torch.allclose(obs[0][:, row, col].cpu(), torch.tensor(AGENT_COLORS[0]).expand(4, 3), atol=1e-5)


def test_agent_is_a_circle_drawn_as_a_box():
    from vmas.simulator.core import Sphere

    cfg = WorldConfig(camera=CAMERA)
    env = make_env(num_envs=1, config=cfg)
    assert isinstance(env.agents[0].shape, Sphere)
    assert env.scenario._visual_size["agent_0"] == cfg.agent_size


@pytest.mark.parametrize("obstacle", [0, 1])  # a box and a cylinder
def test_full_speed_agent_does_not_pass_a_static_obstacle(obstacle):
    cfg = WorldConfig(n_obstacles=2, camera=CAMERA)
    env = make_env(num_envs=1, device="cpu", config=cfg)
    env.reset()
    for o in env.scenario._obstacles:
        o.set_pos(torch.tensor([[50.0, 50.0]]), batch_index=None)
    env.scenario._obstacles[obstacle].set_pos(torch.zeros(1, 2), batch_index=None)
    agent = env.agents[0]
    agent.set_pos(torch.tensor([[-0.6, 0.0]]), batch_index=None)
    agent.set_rot(torch.zeros(1, 1), batch_index=None)
    for _ in range(40):
        env.step([torch.tensor([[1.0, 0.0]])])
        assert agent.state.pos[0, 0] < 0.0


def test_full_speed_agent_does_not_cross_a_slanted_wall():
    import math

    from vmas.simulator.core import Box, Landmark

    class Slanted(Scenario):
        def make_world(self, batch_dim, device, **kwargs):
            world = super().make_world(batch_dim, device, **kwargs)
            self.slant = Landmark(name="slant", shape=Box(8.0, 0.1), movable=False, collide=True)
            world.add_landmark(self.slant)
            return world

    import vmas

    phi = math.radians(45)
    env = vmas.make_env(
        scenario=Slanted(), num_envs=1, device="cpu",
        config=WorldConfig(n_obstacles=0, camera=CAMERA),
    )
    env.reset()
    env.scenario.slant.set_rot(torch.full((1, 1), phi), batch_index=None)
    env.scenario.slant.set_pos(torch.zeros(1, 2), batch_index=None)
    agent = env.agents[0]
    agent.set_pos(torch.tensor([[-0.8, 0.0]]), batch_index=None)
    agent.set_rot(torch.zeros(1, 1), batch_index=None)
    normal = torch.tensor([-math.sin(phi), math.cos(phi)])
    side = torch.sign(torch.tensor([-0.8, 0.0]) @ normal)
    for _ in range(40):
        env.step([torch.tensor([[1.0, 0.0]])])
        assert (agent.state.pos[0] @ normal) * side > 0
