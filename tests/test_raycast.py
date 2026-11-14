import math

import torch

from agent_world.raycast import Scene, intersect, render


def make_scene(entities):
    """entities: list of (x, y, rot, half_x, half_y, height, is_box)."""
    e = torch.tensor([row[:6] for row in entities], dtype=torch.float32)
    return Scene(
        pos=e[None, :, :2],
        rot=e[None, :, 2],
        half_size=e[:, 3:5],
        height=e[:, 5],
        is_box=torch.tensor([row[6] for row in entities]),
        color=torch.eye(3)[: len(entities)],
    )


def ray(*direction):
    d = torch.tensor(direction, dtype=torch.float32)
    return (d / d.norm())[None, None]


ORIGIN = torch.tensor([[0.0, 0.0, 1.0]])
DOWN_FORWARD = ray(1.0, 0.0, -1.0)  # reaches the ground at x = 1


def test_cylinder_top_and_side():
    scene = make_scene([(0.0, 0.0, 0.0, 0.2, 0.2, 0.4, False)])
    # Straight down onto the top disc.
    assert math.isclose(intersect(ORIGIN, ray(0.0, 0.0, -1.0), scene).item(), 0.6, abs_tol=1e-5)
    # From the side: a cylinder of radius 0.2 at x = 2, nearly level ray at z ~ 0.2.
    scene = make_scene([(2.0, 0.0, 0.0, 0.2, 0.2, 0.4, False)])
    origin = torch.tensor([[0.0, 0.0, 0.2]])
    assert math.isclose(intersect(origin, ray(1.0, 0.0, -1e-4), scene).item(), 1.8, abs_tol=1e-3)


def test_box_respects_rotation_and_height():
    # A long thin box across the ray's path only when rotated 90 degrees.
    along = make_scene([(1.0, 0.3, 0.0, 0.5, 0.05, 2.0, True)])
    across = make_scene([(1.0, 0.3, math.pi / 2, 0.5, 0.05, 2.0, True)])
    assert intersect(ORIGIN, DOWN_FORWARD, along).isinf().all()
    assert intersect(ORIGIN, DOWN_FORWARD, across).isfinite().all()
    # A box lower than where the ray passes is missed.
    low = make_scene([(0.5, 0.0, 0.0, 0.05, 0.05, 0.1, True)])
    assert intersect(ORIGIN, DOWN_FORWARD, low).isinf().all()


def test_nearest_entity_occludes_and_mask_hides():
    scene = make_scene(
        [
            (0.5, 0.0, 0.0, 0.05, 0.5, 2.0, True),  # red wall, near
            (0.8, 0.0, 0.0, 0.1, 0.1, 2.0, False),  # green cylinder, behind it
        ]
    )
    rgb, depth = render(ORIGIN, DOWN_FORWARD, scene, max_range=10.0)
    assert rgb[0, 0].argmax() == 0
    assert math.isclose(depth.item(), 0.45 * math.sqrt(2), abs_tol=1e-4)

    hidden_wall = torch.tensor([False, True])
    rgb, _ = render(ORIGIN, DOWN_FORWARD, scene, max_range=10.0, visible=hidden_wall)
    assert rgb[0, 0].argmax() == 1


def test_ground_and_background():
    empty = make_scene([(50.0, 50.0, 0.0, 0.1, 0.1, 0.1, True)])
    rgb, depth = render(ORIGIN, DOWN_FORWARD, empty, max_range=10.0)
    assert math.isclose(depth.item(), math.sqrt(2), abs_tol=1e-5)
    rgb_far, depth_far = render(ORIGIN, ray(1.0, 0.0, -0.01), empty, max_range=10.0)
    assert depth_far.item() == 10.0
    assert not torch.allclose(rgb, rgb_far)
