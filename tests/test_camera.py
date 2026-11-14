import math

import torch

from agent_world.camera import CameraConfig, local_ray_directions, world_rays


def test_rays_are_unit_and_look_down():
    cfg = CameraConfig()
    dirs = local_ray_directions(cfg)
    assert dirs.shape == (cfg.height * cfg.width, 3)
    assert torch.allclose(dirs.norm(dim=-1), torch.ones(len(dirs)), atol=1e-5)
    assert (dirs[:, 2] < 0).all()  # nothing above the horizon


def test_view_bounds():
    cfg = CameraConfig(width=201, height=2001)
    dirs = local_ray_directions(cfg).reshape(cfg.height, cfg.width, 3)
    centre = cfg.width // 2

    # Top edge: the horizon, straight ahead.
    top = dirs[0, centre]
    assert top[0] > 0.99 and abs(top[2]) < 2e-3

    # Bottom edge meets the ground `rear_distance` behind the agent.
    bottom = dirs[-1, centre]
    ground_x = bottom[0] * (cfg.eye_height / -bottom[2])
    assert math.isclose(ground_x, -cfg.rear_distance, abs_tol=2e-3)

    # Left/right edges of the middle row are +-half_fov off the optical axis.
    pitch = cfg.phi_max / 2
    axis = torch.tensor([math.cos(pitch), 0.0, -math.sin(pitch)])
    for col, sign in ((0, 1), (-1, -1)):
        ray = dirs[cfg.height // 2, col]
        angle = math.degrees(math.acos(float(ray @ axis)))
        assert math.isclose(angle, cfg.half_fov_deg, abs_tol=0.5)
        assert sign * ray[1] > 0  # first column looks left


def test_world_rays_follow_heading():
    cfg = CameraConfig(eye_back=0.5)
    dirs = local_ray_directions(cfg)
    pos = torch.tensor([[1.0, 2.0]])
    heading = torch.tensor([math.pi / 2])  # facing +y
    origin, world = world_rays(cfg, dirs, pos, heading)
    assert torch.allclose(origin, torch.tensor([[1.0, 1.5, cfg.eye_height]]), atol=1e-6)
    # Forward component of the local ray becomes +y in the world.
    assert torch.allclose(world[0, :, 1], dirs[:, 0], atol=1e-6)
    assert torch.allclose(world[0, :, 2], dirs[:, 2], atol=1e-6)
