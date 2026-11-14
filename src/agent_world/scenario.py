"""The world, as a VMAS scenario.

A walled square room holding a few static obstacles and one or more agents.
Agents steer like a vehicle (forward speed, turn rate) and see the world
through a third-person camera (see :mod:`agent_world.camera`).

There is no task yet: the reward is zero. This file only wires entities,
physics and observation together; the camera and ray casting live in their own
modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import torch
from torch import Tensor
from vmas.simulator.core import Agent, Box, Entity, Landmark, Sphere, World
from vmas.simulator.dynamics.diff_drive import DiffDrive
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import ScenarioUtils

from agent_world.camera import CameraConfig, local_ray_directions, world_rays
from agent_world.raycast import Palette, Scene, render

RGB = tuple[float, float, float]

AGENT_COLORS: list[RGB] = [
    (0.20, 0.45, 0.85),
    (0.85, 0.30, 0.25),
    (0.25, 0.70, 0.35),
    (0.80, 0.55, 0.15),
]
OBSTACLE_COLORS: list[RGB] = [
    (0.60, 0.35, 0.65),
    (0.90, 0.75, 0.25),
    (0.30, 0.65, 0.65),
    (0.75, 0.45, 0.30),
]
WALL_COLOR: RGB = (0.50, 0.46, 0.42)


@dataclass(frozen=True)
class WorldConfig:
    n_agents: int = 1
    arena_size: float = 4.0
    """Side length of the square room."""
    n_obstacles: int = 6

    agent_radius: float = 0.1
    agent_height: float = 0.25
    max_speed: float = 1.0
    """Forward speed at full action, in world units per second."""
    max_turn_rate: float = 2.0
    """Turn rate at full action, in radians per second."""

    wall_height: float = 1.0
    wall_thickness: float = 0.1
    obstacle_size: float = 0.3
    """Typical obstacle footprint (box side, or cylinder diameter)."""
    obstacle_heights: tuple[float, ...] = (0.15, 0.4, 0.8)
    """Obstacles cycle through these heights; some are lower than the eye."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    palette: Palette = field(default_factory=Palette)


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        """Build the world. Pass ``config=WorldConfig(...)`` or its fields as kwargs."""
        cfg = kwargs.pop("config", None)
        if cfg is None:
            names = {f.name for f in fields(WorldConfig)}
            cfg = WorldConfig(**{k: kwargs.pop(k) for k in list(kwargs) if k in names})
        ScenarioUtils.check_kwargs_consumed(kwargs)
        self.cfg: WorldConfig = cfg

        # Hard outer bounds, just inside the walls, so nothing can leave the room.
        bound = cfg.arena_size / 2 - cfg.wall_thickness / 2 - cfg.agent_radius
        world = World(batch_dim, device, x_semidim=bound, y_semidim=bound)
        # (entity, height) for everything that can be seen, in a fixed order.
        self._visible: list[tuple[Entity, float]] = []

        for i in range(cfg.n_agents):
            self._add_agent(world, f"agent_{i}", AGENT_COLORS[i % len(AGENT_COLORS)])

        self._walls = [
            self._add_landmark(
                world,
                f"wall_{i}",
                Box(cfg.arena_size + cfg.wall_thickness, cfg.wall_thickness),
                cfg.wall_height,
                WALL_COLOR,
            )
            for i in range(4)
        ]
        self._obstacles = []
        for i in range(cfg.n_obstacles):
            size = cfg.obstacle_size
            shape = Box(size, size) if i % 2 == 0 else Sphere(size / 2)
            height = cfg.obstacle_heights[i % len(cfg.obstacle_heights)]
            color = OBSTACLE_COLORS[i % len(OBSTACLE_COLORS)]
            self._obstacles.append(
                self._add_landmark(world, f"obstacle_{i}", shape, height, color)
            )

        self._build_static_scene(device)
        self._local_dirs = local_ray_directions(cfg.camera, device)
        return world

    def _add_agent(self, world: World, name: str, color: RGB) -> None:
        cfg = self.cfg
        agent = Agent(
            name=name,
            shape=Sphere(cfg.agent_radius),
            dynamics=DiffDrive(world),
            u_range=[1.0, 1.0],
            u_multiplier=[cfg.max_speed, cfg.max_turn_rate],
            color=color,
        )
        world.add_agent(agent)
        self._visible.append((agent, cfg.agent_height))

    def _add_landmark(
        self, world: World, name: str, shape, height: float, color: RGB
    ) -> Landmark:
        landmark = Landmark(name=name, shape=shape, collide=True, color=color)
        world.add_landmark(landmark)
        self._visible.append((landmark, height))
        return landmark

    def _build_static_scene(self, device) -> None:
        """Per-entity quantities that never change: size, height, colour."""
        half_size, is_box = [], []
        for entity, _ in self._visible:
            shape = entity.shape
            if isinstance(shape, Box):
                half_size.append((shape.length / 2, shape.width / 2))
                is_box.append(True)
            elif isinstance(shape, Sphere):
                half_size.append((shape.radius, shape.radius))
                is_box.append(False)
            else:
                raise NotImplementedError(f"No visual model for {type(shape).__name__}")
        self._half_size = torch.tensor(half_size, device=device)
        self._is_box = torch.tensor(is_box, device=device)
        self._height = torch.tensor([h for _, h in self._visible], device=device)
        self._color = torch.tensor(
            [
                e.color.value if hasattr(e.color, "value") else e.color
                for e, _ in self._visible
            ],
            device=device,
        )

    def reset_world_at(self, env_index: int | None = None) -> None:
        cfg = self.cfg
        half = cfg.arena_size / 2
        n = self.world.batch_dim if env_index is None else 1

        # Walls: bottom, top (along x), then left, right (rotated 90 degrees).
        wall_poses = [
            ((0.0, -half), 0.0),
            ((0.0, half), 0.0),
            ((-half, 0.0), torch.pi / 2),
            ((half, 0.0), torch.pi / 2),
        ]
        for wall, (pos, rot) in zip(self._walls, wall_poses):
            wall.set_pos(
                torch.tensor(pos, device=self.world.device).expand(n, 2),
                batch_index=env_index,
            )
            wall.set_rot(
                torch.full((n, 1), rot, device=self.world.device),
                batch_index=env_index,
            )

        margin = cfg.wall_thickness / 2 + cfg.obstacle_size
        bounds = (-half + margin, half - margin)
        ScenarioUtils.spawn_entities_randomly(
            self._obstacles + self.world.agents,
            self.world,
            env_index,
            min_dist_between_entities=cfg.obstacle_size + 2 * cfg.agent_radius + 0.1,
            x_bounds=bounds,
            y_bounds=bounds,
        )
        for agent in self.world.agents:
            heading = (torch.rand(n, 1, device=self.world.device) * 2 - 1) * torch.pi
            agent.set_rot(heading, batch_index=env_index)

    def scene(self) -> Scene:
        """Snapshot of everything visible, ready for ray casting."""
        return Scene(
            pos=torch.stack([e.state.pos for e, _ in self._visible], dim=1),
            rot=torch.stack([e.state.rot.squeeze(-1) for e, _ in self._visible], dim=1),
            half_size=self._half_size,
            height=self._height,
            is_box=self._is_box,
            color=self._color,
        )

    def view(self, agent: Agent) -> tuple[Tensor, Tensor]:
        """What ``agent`` sees: rgb [B, H, W, 3] in [0, 1] and depth [B, H, W]."""
        cam = self.cfg.camera
        origin, dirs = world_rays(
            cam, self._local_dirs, agent.state.pos, agent.state.rot.squeeze(-1)
        )
        rgb, depth = render(origin, dirs, self.scene(), cam.max_range, self.cfg.palette)
        batch = rgb.shape[0]
        return (
            rgb.reshape(batch, cam.height, cam.width, 3),
            depth.reshape(batch, cam.height, cam.width),
        )

    def observation(self, agent: Agent) -> Tensor:
        return self.view(agent)[0]

    def reward(self, agent: Agent) -> Tensor:
        return torch.zeros(self.world.batch_dim, device=self.world.device)
