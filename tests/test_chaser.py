import math

import torch

from agent_world import CameraConfig, WorldConfig, make_env

CAMERA = CameraConfig(width=48, height=32)


def setup(num_envs=1, **kw):
    kw = {"n_obstacles": 0, "rotating_patches": False, "camera": CAMERA, **kw}
    env = make_env(num_envs=num_envs, device="cpu", seed=0, config=WorldConfig(**kw))
    env.reset()
    for i in range(len(env.scenario._chasers.landmarks)):
        put(env, i, 7.0, 3.0 - i * 0.3)  # park out of the way
    return env, env.scenario._chasers


def put(env, i, x, y, heading=0.0, chaser=True):
    ent = env.scenario._chasers.landmarks[i] if chaser else env.agents[i]
    n = env.num_envs
    ent.set_pos(torch.tensor([[x, y]], device=env.device).expand(n, 2), batch_index=None)
    ent.set_rot(torch.full((n, 1), heading, device=env.device), batch_index=None)
    ent.set_vel(torch.zeros(n, 2, device=env.device), batch_index=None)
    if chaser:
        env.scenario._chasers.heading[:, i] = heading


def step(env, n=1):
    for _ in range(n):
        env.step([torch.zeros(env.num_envs, 2, device=env.device) for _ in env.agents])


def test_chasers_are_landmarks_not_agents():
    env, ch = setup()
    assert len(ch.landmarks) == 4 and len(env.agents) == 1


def test_speed_is_seventy_percent_of_max():
    env, ch = setup()
    put(env, 0, -5.0, 0.0), put(env, 0, -3.0, 0.0, chaser=False)
    step(env, 3)
    assert math.isclose(ch.landmarks[0].state.vel.norm().item(), 0.7, abs_tol=1e-3)


def test_stops_when_nothing_is_seen():
    env, ch = setup()
    put(env, 0, -3.0, 0.0), put(env, 0, -3.0, -4.0, chaser=False)  # too far, below
    ch.landmarks[0].set_vel(torch.ones(1, 2), batch_index=None)
    step(env, 2)
    pos = ch.landmarks[0].state.pos.clone()
    step(env, 5)
    assert ch.landmarks[0].state.vel.norm() < 1e-4
    assert torch.allclose(ch.landmarks[0].state.pos, pos, atol=1e-4)


def moved(env, ch, agent_xy, heading=0.0, chaser_xy=(-5.0, 0.0)):
    put(env, 0, *chaser_xy, heading)
    put(env, 0, *agent_xy, chaser=False)
    start = ch.landmarks[0].state.pos.clone()
    step(env, 3)
    return (ch.landmarks[0].state.pos - start).norm().item() > 0.05


def test_sees_in_front_but_not_behind():
    env, ch = setup()
    assert moved(env, ch, (-3.0, 0.0))
    assert not moved(env, ch, (-7.0, 0.0))
    assert not moved(env, ch, (-5.0, 2.0))  # 90 degrees off, outside the field of view


def test_does_not_see_through_a_wall():
    env, ch = setup()
    assert not moved(env, ch, (1.0, 3.0), chaser_xy=(-1.0, 3.0))
    assert moved(env, ch, (1.0, 0.0), chaser_xy=(-1.0, 0.0))  # through the doorway


def test_chases_and_closes_the_distance():
    env, ch = setup()
    put(env, 0, -5.0, 0.0), put(env, 0, -2.0, 2.0, chaser=False)
    gap = lambda: (ch.landmarks[0].state.pos - env.agents[0].state.pos).norm().item()
    before = gap()
    step(env, 15)
    assert gap() < before - 0.5


def test_a_patch_turns_a_chaser_that_sees_nothing():
    env, ch = setup(rotating_patches=True)
    put(env, 0, 0.0, 0.0), put(env, 0, -7.0, -3.0, chaser=False)
    step(env, 5)
    assert -1.4 < ch.heading[0, 0].item() < -1.1  # 2.5 rad/s for 0.5 s


def hits(env, ch, steps):
    times = []
    for t in range(steps):
        step(env)
        if ch.touched[0].item():
            times.append(t)
    return times


def test_bump_displacement_and_flag():
    env, ch = setup()
    put(env, 0, -3.19, 0.0, math.pi), put(env, 0, -3.0, 0.0, chaser=False)  # facing away, touching
    start = env.agents[0].state.pos.clone()
    step(env)
    assert ch.touched[0].item()
    step(env, 2)
    d = (env.agents[0].state.pos - start)[0]
    assert math.isclose(d[0].item(), 0.5, abs_tol=0.15) and abs(d[1].item()) < 0.15
    step(env)
    assert not ch.touched[0].item()


def test_bump_cooldown_spaces_hits_by_a_second():
    env, ch = setup()
    put(env, 0, -3.2, 0.0), put(env, 0, -3.0, 0.0, chaser=False)
    t = hits(env, ch, 30)
    assert len(t) >= 2 and 0.9 <= (t[1] - t[0]) * 0.1 <= 1.4


def test_ghost_is_ignored():
    env, ch = setup(spectator=True)
    put(env, 0, -5.0, 0.0), put(env, 0, -7.0, 3.0, chaser=False)
    put(env, 1, -4.9, 0.0, chaser=False)  # the ghost, on top of the chaser and ahead of it
    step(env, 5)
    assert ch.landmarks[0].state.vel.norm() < 1e-4
    assert not ch.touched.any()
    assert torch.allclose(env.agents[1].state.pos, torch.tensor([[-4.9, 0.0]]), atol=1e-4)


def test_room_zero_chasers_spawn_far_from_agents():
    env = make_env(num_envs=64, device="cpu", seed=1, config=WorldConfig(n_agents=1, camera=CAMERA))
    for _ in range(3):
        env.reset()
        for c in env.scenario._chasers.landmarks[:2]:
            for a in env.agents:
                assert (c.state.pos - a.state.pos).norm(dim=-1).min() >= 4.0
        for c in env.scenario._chasers.landmarks[2:]:
            assert (c.state.pos[:, 0] > 0).all()


def test_reset_one_world_resets_its_chasers():
    env, ch = setup(num_envs=2)
    ch.cooldown[:] = 1.0
    env.reset_at(1)
    assert ch.cooldown[1].sum() == 0 and ch.cooldown[0].sum() == 4.0


def test_chasers_are_striped_red_and_black():
    env, ch = setup()
    put(env, 0, -3.0, 0.0, chaser=False)
    put(env, 0, -2.0, 0.0, math.pi / 2)  # side on, bands along its length
    rgb = env.scenario.view(env.agents[0])[0][0].reshape(-1, 3)
    assert ((rgb[:, 0] > 0.5) & (rgb[:, 1] < 0.2)).any()
    assert (rgb.amax(-1) < 0.1).any()
