import torch

from agent_world import CameraConfig, WorldConfig, make_env
from agent_world.scenario import AGENT_COLORS

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
        expected = torch.tensor(AGENT_COLORS[i]).expand(4, 3)
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
    cfg = WorldConfig(n_agents=3, camera=CAMERA)
    env = make_env(num_envs=8, seed=1, config=cfg)
    env.reset()
    for _ in range(150):
        env.step([torch.tensor([[1.0, 0.0]]).expand(8, 2) for _ in env.agents])
    limit = cfg.arena_size / 2 - cfg.wall_thickness / 2 - cfg.agent_radius
    for agent in env.agents:
        assert agent.state.pos.abs().max() <= limit + 1e-2
