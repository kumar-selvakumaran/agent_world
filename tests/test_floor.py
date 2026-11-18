import math

import numpy as np
import pytest
import torch

from agent_world import CameraConfig, WorldConfig, make_env
from agent_world.floor import ARENA_AREA, SIDE, TEXEL, Floor, build_map, place_sections

CAMERA = CameraConfig(width=64, height=48)


def axis_alignment(image):
    """1 when every edge in the [H, W, 3] image lies along the image axes, 0 when none does."""
    gray = image.mean(-1)
    gy, gx = np.gradient(gray)
    magnitude, angle = np.hypot(gx, gy), np.arctan2(gy, gx)
    return float(abs((magnitude * np.exp(4j * angle)).sum()) / magnitude.sum())


def boxes(sections):
    out = []
    for cx, cy, w, h, angle in sections:
        bw = w * abs(math.cos(angle)) + h * abs(math.sin(angle))
        bh = w * abs(math.sin(angle)) + h * abs(math.cos(angle))
        out.append((cx - bw / 2, cx + bw / 2, cy - bh / 2, cy + bh / 2))
    return out


def test_sections_reach_the_coverage_without_overlapping():
    sections, coverage = place_sections(0, 0.45)
    assert coverage >= 0.45
    for a, (x0, x1, y0, y1) in enumerate(boxes(sections)):
        assert x0 >= 0 and y0 >= 0 and x1 <= SIDE and y1 <= SIDE
        for x2, x3, y2, y3 in boxes(sections)[a + 1 :]:
            assert x1 <= x2 or x3 <= x0 or y1 <= y2 or y3 <= y0  # bounding boxes are disjoint
    for _, _, w, h, _ in sections:
        assert 0.03 * ARENA_AREA - 1e-6 <= w * h <= 0.10 * ARENA_AREA + 1e-6


def test_the_map_is_deterministic_and_seeds_differ():
    first = build_map.__wrapped__(0)  # bypass the cache
    assert first.shape == (int(SIDE / TEXEL), int(SIDE / TEXEL), 3) and first.dtype == torch.uint8
    assert torch.equal(first, build_map(0))
    assert not torch.equal(first, build_map(1))


def test_the_floor_has_no_preferred_direction():
    floor = Floor.create(1, "cpu", reach=9.0)
    xs, ys = torch.linspace(-8, 8, 320), torch.linspace(-4, 4, 160)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    window = floor.colour(torch.stack([xx, yy], -1).reshape(1, -1, 2)).reshape(160, 320, 3).numpy()
    checker = ((np.floor(np.stack([xx, yy], -1) / 0.5).sum(-1) % 2 == 0)[..., None] * np.ones(3)).astype(float)
    assert axis_alignment(checker) > 0.5  # the old floor points along the axes
    assert axis_alignment(window) < 0.15


def test_every_world_gets_its_own_window():
    floor = Floor.create(8, "cpu", reach=9.0)
    assert len({tuple(row) for row in floor.offset.tolist()}) == 8
    assert (floor.offset >= 9.0).all() and (floor.offset <= SIDE - 9.0).all()
    before = floor.offset.clone()
    floor.reseed(3)
    changed = (floor.offset != before).any(-1)
    assert changed.tolist() == [False, False, False, True, False, False, False, False]
    colour = floor.colour(torch.zeros(8, 5, 2))
    assert colour.shape == (8, 5, 3) and colour.min() >= 0 and colour.max() <= 1


def test_the_floor_option():
    assert make_env(num_envs=1, device="cpu", config=WorldConfig(floor="checker")).scenario._floor is None
    assert make_env(num_envs=1, device="cpu", config=WorldConfig()).scenario._floor is not None
    with pytest.raises(ValueError):
        make_env(num_envs=1, device="cpu", config=WorldConfig(floor="sand"))


def motion(floor):
    config = WorldConfig(
        floor=floor, n_obstacles=0, chasers_per_room=0, blocks_per_room=0, food_per_room=0,
        rotating_patches=False, camera=CAMERA,
    )
    env = make_env(num_envs=1, device="cpu", seed=3, config=config)
    env.reset()
    agent, changes = env.agents[0], []
    for x, y, heading in [(-6.0, -1.0, 0.0), (-4.0, 2.0, 0.5), (-6.5, 1.0, -0.4), (-3.0, -2.5, 1.0), (-5.0, 0.5, 2.0)]:
        agent.set_pos(torch.tensor([[x, y]]), batch_index=None)
        agent.set_rot(torch.full((1, 1), heading), batch_index=None)
        agent.state.vel[:] = 0
        before = env.step([torch.zeros(1, 2)])[0][0][0]
        after = env.step([torch.tensor([[1.0, 0.0]])])[0][0][0]
        changes.append((after - before).abs().mean().item())
    return sum(changes) / len(changes)


def test_motion_is_nearly_as_visible_as_on_the_checkerboard():
    assert motion("map") >= 0.7 * motion("checker")
