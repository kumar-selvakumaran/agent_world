import math

import torch

from agent_world.patch import Patch, inside, paint

RING = 0.6  # default ring radius


def at(r, angle):
    return torch.tensor([r * math.cos(angle), r * math.sin(angle)])


def colour_at(point, patch=Patch(center=(0.0, 0.0))):
    base = torch.full((3,), 0.5)
    return paint(point, patch, base)


def test_inside_is_the_disc():
    pos = torch.tensor([[0.5, 0.0], [1.5, 0.0]])
    assert inside(pos, Patch(center=(0.0, 0.0))).tolist() == [True, False]


def test_outside_the_patch_the_ground_is_unchanged():
    assert torch.equal(colour_at(torch.tensor([3.0, 0.0])), torch.full((3,), 0.5))


def test_disc_is_pink_and_the_ring_is_black():
    patch = Patch(center=(0.0, 0.0))
    assert torch.allclose(colour_at(torch.tensor([0.0, 0.0])), torch.tensor(patch.color))
    assert torch.allclose(colour_at(at(RING, math.pi / 4)), torch.tensor(patch.ink))


def test_arrowheads_point_the_way_it_spins():
    # Arrow centred at angle 0. A point off the ring by 0.15 is inside the wide
    # (trailing) end of the arrowhead but outside its narrow (leading) end.
    step = 0.1 / RING
    for rate, trailing in ((-1.0, +step), (1.0, -step)):  # clockwise trails at +angle
        patch = Patch(center=(0.0, 0.0), rate=rate)
        ink = torch.tensor(patch.ink)
        assert torch.allclose(colour_at(at(RING + 0.15, trailing), patch), ink)
        assert not torch.allclose(colour_at(at(RING + 0.15, -trailing), patch), ink)
