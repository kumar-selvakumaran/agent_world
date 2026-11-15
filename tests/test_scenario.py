import pytest
import torch

from agent_world import CameraConfig, Scenario, WorldConfig, make_env
from agent_world.patch import inside
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
    margin = cfg.wall_thickness / 2 + cfg.agent_radius
    limit = torch.tensor([cfg.arena_width / 2 - margin, cfg.arena_height / 2 - margin], device=env.device)
    for agent in env.agents:
        assert (agent.state.pos.abs() <= limit + 1e-2).all()


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
    cfg = WorldConfig(n_obstacles=2, rotating_patches=False, chasers_per_room=0, blocks_per_room=0, camera=CAMERA)
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
        config=WorldConfig(n_obstacles=0, rotating_patches=False, chasers_per_room=0, blocks_per_room=0, camera=CAMERA),
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


def test_agents_start_in_the_first_room():
    env = make_env(num_envs=8, seed=3, config=WorldConfig(n_agents=3, camera=CAMERA))
    env.reset()
    for agent in env.agents:
        assert (agent.state.pos[:, 0] < 0).all()


def test_full_speed_agent_passes_the_doorway_but_not_the_wall():
    env = make_env(num_envs=2, device="cpu", config=WorldConfig(n_obstacles=0, rotating_patches=False, chasers_per_room=0, blocks_per_room=0, camera=CAMERA))
    env.reset()
    agent = env.agents[0]
    agent.set_pos(torch.tensor([[-1.0, 0.0], [-1.0, 2.0]]), batch_index=None)  # doorway, wall
    agent.set_rot(torch.zeros(2, 1), batch_index=None)
    for _ in range(60):
        env.step([torch.tensor([[1.0, 0.0]]).expand(2, 2)])
    assert agent.state.pos[0, 0] > 1.0
    assert agent.state.pos[1, 0] < 0.0


def patch_env(num_envs=1):
    env = make_env(num_envs=num_envs, device="cpu", config=WorldConfig(n_obstacles=0, chasers_per_room=0, blocks_per_room=0, camera=CAMERA))
    env.reset()
    return env, env.scenario._patches[0]


def place(env, x, y, heading=0.0):
    agent = env.agents[0]
    n = agent.state.pos.shape[0]
    agent.set_pos(torch.tensor([[x, y]]).expand(n, 2), batch_index=None)
    agent.set_rot(torch.full((n, 1), heading), batch_index=None)
    return agent


def test_a_patch_turns_an_idle_agent_clockwise():
    env, patch = patch_env()
    agent = place(env, *patch.center)
    for _ in range(5):  # half a second
        env.step([torch.zeros(1, 2)])
    assert -1.3 < agent.state.rot.item() < -0.6


def test_counter_steering_cannot_turn_the_agent_anti_clockwise_on_the_patch():
    env, patch = patch_env()
    agent = place(env, *patch.center)
    headings = []
    for _ in range(20):
        env.step([torch.tensor([[0.0, 1.0]])])  # full counter-steer
        headings.append(agent.state.rot.item())
    assert max(headings) <= 1e-3 and headings[-1] < -0.5  # only slowed


def test_the_patch_acts_only_inside():
    env, patch = patch_env()
    agent = place(env, patch.center[0] + 3.0, 0.0)
    for _ in range(10):
        env.step([torch.zeros(1, 2)])
    assert agent.state.rot.abs().item() < 1e-3


def test_the_doorway_is_crossed_by_turning_before_the_patch_and_slowing_the_spin():
    env, _ = patch_env()
    agent = place(env, -0.95, -0.9)  # just off the patch, where steering is free
    for _ in range(45):
        env.step([torch.tensor([[1.0, 1.0]])])
    assert agent.state.pos[0, 0] > 1.0  # through the doorway into the second room


def test_without_steering_the_patch_keeps_an_agent_circling():
    env, patch = patch_env()
    agent = place(env, patch.center[0] - 1.5, 0.0)
    for _ in range(60):
        env.step([torch.tensor([[1.0, 0.0]])])
    assert agent.state.pos[0, 0] < 1.0


def test_the_patch_is_visible_to_the_agent():
    env, patch = patch_env()
    place(env, patch.center[0] - 3.0, 0.0)
    obs = env.step([torch.zeros(1, 2)])[0][0][0]
    pink = torch.tensor(patch.color)
    assert ((obs - pink).abs().amax(-1) < 0.02).any()


def test_chasers_are_counted_per_room_and_can_be_turned_off():
    env = make_env(num_envs=2, device="cpu", config=WorldConfig(chasers_per_room=3, camera=CAMERA))
    env.reset()
    assert len(env.scenario._chasers.landmarks) == 6 and len(env.agents) == 1
    env = make_env(num_envs=2, device="cpu", config=WorldConfig(chasers_per_room=0, blocks_per_room=0, camera=CAMERA))
    env.reset()
    env.step([torch.zeros(2, 2)])
    assert not env.scenario._chasers.landmarks


def food_env(**kw):
    cfg = WorldConfig(camera=CAMERA, food_respawn_time=0.5, **kw)
    env = make_env(num_envs=2, seed=0, config=cfg)
    env.reset()
    return env, env.scenario


def still(env):
    return [torch.zeros(2, 2, device=env.device) for _ in env.agents]


def test_agent_eats_food_and_it_respawns_in_same_room():
    env, sc = food_env()
    agent = env.agents[0]
    sc._food_xy[:, 0] = agent.state.pos
    old_room = sc._food_xy[:, 0, 0] > 0
    env.step(still(env))
    assert (sc._food_timer[:, 0] > 0).all()
    assert not sc._food().active[:, 0].any()
    for _ in range(6):
        env.step(still(env))
    assert (sc._food_timer[:, 0] == 0).all()
    assert (sc._food_xy[:, 0] - agent.state.pos).norm(dim=-1).min() > 1e-3
    assert ((sc._food_xy[:, 0, 0] > 0) == old_room).all()


def test_ghost_does_not_eat():
    env, sc = food_env(spectator=True)
    ghost = env.agents[-1]
    sc._food_xy[:, 0] = ghost.state.pos
    env.step(still(env))
    assert (sc._food_timer[:, 0] == 0).all()


def test_reset_restores_food():
    env, sc = food_env()
    sc._food_timer[:] = 3.0
    env.reset_at(0)
    assert (sc._food_timer[0] == 0).all() and (sc._food_timer[1] == 3.0).all()
    env.reset()
    assert (sc._food_timer == 0).all()


def test_food_visible_to_agent():
    env, sc = food_env()
    agent = env.agents[0]
    heading = agent.state.rot
    sc._food_xy[:] = (agent.state.pos + 0.5 * torch.cat((heading.cos(), heading.sin()), -1))[:, None]
    rgb = sc.view(agent)[0]
    ochre = torch.tensor((0.80, 0.58, 0.12), device=env.device)
    assert ((rgb - ochre).abs().sum(-1) < 0.05).flatten(1).any(1).all()


def test_food_and_chasers_can_be_switched_off():
    cfg = WorldConfig(food_per_room=0, chasers_per_room=0, blocks_per_room=0, camera=CAMERA)
    env = make_env(num_envs=2, device="cpu", config=cfg)
    env.reset()
    env.step([torch.zeros(2, 2) for _ in env.agents])
