import torch

from agent_world import food
from agent_world.food import Food


def test_paint_colour_and_radius():
    xy = torch.tensor([[[0.0, 0.0], [0.05, 0.0], [0.2, 0.0]]])
    f = Food(torch.zeros(1, 1, 2), torch.ones(1, 1, dtype=torch.bool), 0.08)
    out = food.paint(xy, f, torch.zeros(1, 3, 3))
    assert torch.allclose(out[0, 0], torch.tensor(food.COLOR))
    assert torch.allclose(out[0, 1], torch.tensor(food.COLOR))
    assert (out[0, 2] == 0).all()


def test_inactive_food_not_painted():
    f = Food(torch.zeros(1, 1, 2), torch.zeros(1, 1, dtype=torch.bool), 0.08)
    assert (food.paint(torch.zeros(1, 1, 2), f, torch.zeros(1, 1, 3)) == 0).all()


def test_sample_stays_in_own_room():
    xy = food.sample(50, 4, 2, 8.0, 0.25, "cpu")
    assert xy.shape == (50, 8, 2)
    assert (xy[:, :4, 0] >= -8 + 0.25).all() and (xy[:, :4, 0] <= -0.25).all()
    assert (xy[:, 4:, 0] >= 0.25).all() and (xy[:, 4:, 0] <= 8 - 0.25).all()
    assert (xy[..., 1].abs() <= 4 - 0.25).all()


def test_step_eats_then_respawns():
    xy = torch.zeros(2, 1, 2)
    timer = torch.zeros(2, 1)
    pos = torch.tensor([[[0.1, 0.0]], [[2.0, 0.0]]])
    new = torch.full((2, 1, 2), 3.0)
    args = (pos, torch.tensor([True]), 0.18, 1.0, 0.5, new)
    food.step(xy, timer, *args)
    assert timer[0, 0] == 1.0 and timer[1, 0] == 0.0
    food.step(xy, timer, *args)
    assert timer[0, 0] == 0.5 and (xy[0] == 0).all()
    pos[0] = 9.0
    food.step(xy, timer, *args)
    assert timer[0, 0] == 0.0 and (xy[0] == 3.0).all() and (xy[1] == 0).all()
