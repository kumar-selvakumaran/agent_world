import torch

from agent_world import CameraConfig, WorldConfig, make_env
from agent_world.blocks import touching

CAMERA = CameraConfig(width=16, height=12)


def setup(num_envs=4, **kw):
    cfg = WorldConfig(rotating_patches=False, camera=CAMERA, **kw)
    env = make_env(num_envs=num_envs, device="cpu", seed=0, config=cfg)
    env.reset()
    return env, env.scenario


def put(env, ent, x, y):
    n = env.num_envs
    ent.set_pos(torch.tensor([[x, y]], device=env.device).expand(n, 2), batch_index=None)
    ent.set_vel(torch.zeros(n, 2, device=env.device), batch_index=None)


def step(env, n=1):
    for _ in range(n):
        env.step([torch.zeros(env.num_envs, 2, device=env.device) for _ in env.agents])


def block_xy(sc):
    return torch.stack([b.state.pos for b in sc._blocks], 1)


def test_one_block_per_strip_clear_of_doorway_and_walls():
    cfg = WorldConfig()
    env, sc = setup(num_envs=64)
    pos = block_xy(sc)
    assert pos.shape == (64, 6, 2)
    strip = cfg.room_size / cfg.blocks_per_room
    edges = -cfg.arena_width / 2 + torch.arange(7) * strip
    x = pos[..., 0]
    assert ((x > edges[:-1]) & (x < edges[1:])).all()
    assert (pos[..., 1].abs() < cfg.room_size / 2 - cfg.block_size / 2).all()
    assert (torch.hypot(x, pos[..., 1]) > cfg.patch_radius + 0.3).all()
    assert x.min() > -cfg.arena_width / 2 + cfg.block_size / 2


def test_blocks_can_be_switched_off():
    env, sc = setup(blocks_per_room=0)
    assert sc._blocks == []


def test_obstacles_and_agents_start_clear_of_blocks():
    env, sc = setup(num_envs=16)
    others = torch.stack([e.state.pos for e in sc._obstacles + env.agents], 1)
    gap = ((others[:, :, None] - block_xy(sc)[:, None]).abs() - 0.3).clamp(min=0).norm(dim=-1)
    assert (gap > 0.1).all()


def test_block_is_solid_for_a_full_speed_agent():
    env, sc = setup(num_envs=1, n_obstacles=0, chasers_per_room=0)
    put(env, sc._blocks[0], -4.0, 0.0)
    put(env, env.agents[0], -5.0, 0.0)
    env.agents[0].set_rot(torch.zeros(1, 1), batch_index=None)
    for _ in range(60):
        env.step([torch.tensor([[1.0, 0.0]])])
    assert env.agents[0].state.pos[0, 0] < -4.0 - 0.3 + 0.1 + 0.05


def test_touching_is_distance_to_the_box():
    blocks = torch.zeros(1, 1, 2)
    pos = torch.tensor([[[0.35, 0.0], [0.45, 0.0], [0.4, 0.4]]])
    assert touching(pos, blocks, 0.3, 0.1).tolist() == [[True, False, False]]


def test_chaser_on_a_block_respawns_far_from_agent_in_its_room():
    env, sc = setup(num_envs=8, n_obstacles=0)
    ch = sc._chasers
    put(env, env.agents[0], -3.0, 0.0)
    put(env, ch.landmarks[0], -4.0, 3.0)
    put(env, sc._blocks[0], -4.0, 3.0)
    ch.landmarks[0].set_vel(torch.ones(8, 2), batch_index=None)
    step(env)
    p = ch.landmarks[0].state.pos
    assert (torch.norm(p - env.agents[0].state.pos, dim=-1) >= 4.0).all()
    assert (p[:, 0] < 0).all()
    assert (ch.cooldown[:, 0] == 0).all()


def test_chaser_off_a_block_stays_put():
    env, sc = setup(num_envs=2, n_obstacles=0)
    ch = sc._chasers
    put(env, env.agents[0], -3.0, 0.0)
    put(env, sc._blocks[0], -4.0, 3.0)
    put(env, ch.landmarks[0], -4.0, 3.0 - 0.3 - 0.1 - 0.1)
    put(env, ch.landmarks[1], 7.0, 3.0)
    before = ch.landmarks[0].state.pos.clone()
    step(env)
    assert torch.allclose(ch.landmarks[0].state.pos, before, atol=1e-3)


def test_a_block_blocks_the_chasers_line_of_sight():
    env, sc = setup(num_envs=1, n_obstacles=0)
    ch = sc._chasers
    for i, c in enumerate(ch.landmarks):
        put(env, c, 7.0, 3.0 - i * 0.3)
    for i, b in enumerate(sc._blocks):
        put(env, b, 7.0, -3.0 + i * 0.7)
    put(env, ch.landmarks[0], -5.0, 0.0)
    put(env, env.agents[0], -3.0, 0.0)
    ch.heading[:, 0] = 0.0
    step(env, 3)
    assert ch.landmarks[0].state.vel.norm() > 0.5
    put(env, sc._blocks[0], -4.0, 0.0)
    put(env, ch.landmarks[0], -5.0, 0.0)
    step(env, 3)
    assert ch.landmarks[0].state.vel.norm() < 1e-4


def test_chaser_passes_through_a_block_without_being_stopped():
    env, sc = setup(num_envs=1, n_obstacles=0)
    assert not sc._chasers.landmarks[0].collides(sc._blocks[0])
    assert env.agents[0].collides(sc._blocks[0])


def test_reset_places_blocks_per_world():
    env, sc = setup(num_envs=3)
    before = block_xy(sc).clone()
    env.reset_at(1)
    after = block_xy(sc)
    assert torch.equal(before[[0, 2]], after[[0, 2]])
    assert not torch.equal(before[1], after[1])
