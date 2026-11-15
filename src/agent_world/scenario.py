"""The world, as a VMAS scenario.

A walled square room holding a few static obstacles and one or more agents.
Agents steer like a vehicle (forward speed, turn rate) and see the world
through a third-person camera (see :mod:`agent_world.camera`).

With ``spectator=True`` one extra agent, the spectator, is added: it moves and
sees exactly like the others, but passes through everything and is invisible
to everyone but itself. It is meant for a human exploring the world.

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
SPECTATOR_COLOR: RGB = (0.95, 0.95, 0.95)
SPECTATOR_NAME = "spectator"


@dataclass(frozen=True)
class WorldConfig:
    n_agents: int = 1
    room_size: float = 8.0
    """Side length of each square room."""
    n_rooms: int = 2
    """Rooms are placed in a row along x, each pair joined by a doorway."""
    doorway_width: float = 1.0
    """Gap in the wall between two rooms, centred on the wall."""
    n_obstacles: int = 12
    """Spread over the rooms, one after the other."""
    spectator: bool = False
    """Add a ghost agent for a human observer. Leave off for training: every
    agent costs one rendered view per step."""

    agent_radius: float = 0.1
    """Radius of the agent's physics body, a circle. Also used for room bounds and spawn spacing."""
    agent_size: tuple[float, float] = (0.16, 0.12)
    """Length (along the heading) and width of the box drawn for the agent, so its
    heading is visible. It fits inside ``agent_radius``; the physics body is the circle."""
    agent_height: float = 0.25
    max_speed: float = 1.0
    """Forward speed at full action, in world units per second."""
    max_turn_rate: float = 2.0
    """Turn rate at full action, in radians per second."""
    collision_force: float = 1000.0
    """Contact stiffness (VMAS defaults to 100). At full speed, 1000 with 5 substeps
    holds agents against walls and obstacles without violent ejections."""
    substeps: int = 5
    """Physics substeps per step. Stiff contacts need them to stay stable."""

    wall_height: float = 1.0
    wall_thickness: float = 0.1
    obstacle_size: float = 0.3
    """Typical obstacle footprint (box side, or cylinder diameter)."""
    obstacle_heights: tuple[float, ...] = (0.15, 0.4, 0.8)
    """Obstacles cycle through these heights; some are lower than the eye."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    palette: Palette = field(default_factory=Palette)

    @property
    def arena_width(self) -> float:
        return self.n_rooms * self.room_size

    @property
    def arena_height(self) -> float:
        return self.room_size


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        """Build the world. Pass ``config=WorldConfig(...)`` or its fields as kwargs."""
        cfg = kwargs.pop("config", None)
        if cfg is None:
            names = {f.name for f in fields(WorldConfig)}
            cfg = WorldConfig(**{k: kwargs.pop(k) for k in list(kwargs) if k in names})
        ScenarioUtils.check_kwargs_consumed(kwargs)
        self.cfg: WorldConfig = cfg

        # Walls stop ordinary agents; these bounds also stop the spectator,
        # which passes through walls, from leaving the room.
        margin = cfg.wall_thickness / 2 + cfg.agent_radius
        world = World(
            batch_dim,
            device,
            x_semidim=cfg.arena_width / 2 - margin,
            y_semidim=cfg.arena_height / 2 - margin,
            collision_force=cfg.collision_force,
            substeps=cfg.substeps,
        )
        # (entity, height) for everything that can be seen, in a fixed order.
        self._visible: list[tuple[Entity, float]] = []
        # Entities drawn as a box of this size instead of their physics shape.
        self._visual_size: dict[str, tuple[float, float]] = {}

        for i in range(cfg.n_agents):
            self._add_agent(world, f"agent_{i}", AGENT_COLORS[i % len(AGENT_COLORS)])
        if cfg.spectator:
            self._add_agent(world, SPECTATOR_NAME, SPECTATOR_COLOR, ghost=True)

        self._wall_specs = self._make_wall_specs()
        self._walls = [
            self._add_landmark(
                world,
                f"wall_{i}",
                Box(length, cfg.wall_thickness),
                cfg.wall_height,
                WALL_COLOR,
            )
            for i, (_, _, length) in enumerate(self._wall_specs)
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

    def _make_wall_specs(self) -> list[tuple[tuple[float, float], float, float]]:
        """Every wall as (centre, rotation, length): the outer walls, then the
        wall between each pair of rooms, split in two around its doorway."""
        cfg = self.cfg
        w, h, t = cfg.arena_width, cfg.arena_height, cfg.wall_thickness
        specs = [
            ((0.0, -h / 2), 0.0, w + t),
            ((0.0, h / 2), 0.0, w + t),
            ((-w / 2, 0.0), torch.pi / 2, h + t),
            ((w / 2, 0.0), torch.pi / 2, h + t),
        ]
        piece = (h - cfg.doorway_width) / 2
        offset = (cfg.doorway_width + piece) / 2
        for k in range(1, cfg.n_rooms):
            x = -w / 2 + k * cfg.room_size
            specs += [((x, -offset), torch.pi / 2, piece), ((x, offset), torch.pi / 2, piece)]
        return specs

    def _add_agent(self, world: World, name: str, color: RGB, ghost: bool = False) -> None:
        cfg = self.cfg
        agent = Agent(
            name=name,
            shape=Sphere(cfg.agent_radius),
            dynamics=DiffDrive(world),
            u_range=[1.0, 1.0],
            u_multiplier=[cfg.max_speed, cfg.max_turn_rate],
            collide=not ghost,
            color=color,
        )
        world.add_agent(agent)
        self._visible.append((agent, cfg.agent_height))
        self._visual_size[name] = cfg.agent_size

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
            if entity.name in self._visual_size:
                length, width = self._visual_size[entity.name]
                half_size.append((length / 2, width / 2))
                is_box.append(True)
                continue
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
        self._index = {e.name: i for i, (e, _) in enumerate(self._visible)}
        self._is_ghost = torch.tensor(
            [e.name == SPECTATOR_NAME for e, _ in self._visible], device=device
        )
        self._color = torch.tensor(
            [
                e.color.value if hasattr(e.color, "value") else e.color
                for e, _ in self._visible
            ],
            device=device,
        )

    def reset_world_at(self, env_index: int | None = None) -> None:
        cfg = self.cfg
        n = self.world.batch_dim if env_index is None else 1

        for wall, (pos, rot, _) in zip(self._walls, self._wall_specs):
            wall.set_pos(
                torch.tensor(pos, device=self.world.device).expand(n, 2),
                batch_index=env_index,
            )
            wall.set_rot(
                torch.full((n, 1), rot, device=self.world.device),
                batch_index=env_index,
            )

        # Agents start in the first room; obstacles are spread over all rooms.
        margin = cfg.wall_thickness / 2 + cfg.obstacle_size
        y_bounds = (-cfg.arena_height / 2 + margin, cfg.arena_height / 2 - margin)
        for room in range(cfg.n_rooms):
            left = -cfg.arena_width / 2 + room * cfg.room_size
            entities = self._obstacles[room :: cfg.n_rooms]
            if room == 0:
                entities = entities + self.world.agents
            if entities:
                ScenarioUtils.spawn_entities_randomly(
                    entities,
                    self.world,
                    env_index,
                    min_dist_between_entities=cfg.obstacle_size + 2 * cfg.agent_radius + 0.1,
                    x_bounds=(left + margin, left + cfg.room_size - margin),
                    y_bounds=y_bounds,
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
        # Ghosts are hidden from everyone except themselves.
        visible = ~self._is_ghost
        visible[self._index[agent.name]] = True
        rgb, depth = render(
            origin, dirs, self.scene(), cam.max_range, self.cfg.palette, visible
        )
        batch = rgb.shape[0]
        return (
            rgb.reshape(batch, cam.height, cam.width, 3),
            depth.reshape(batch, cam.height, cam.width),
        )

    def observation(self, agent: Agent) -> Tensor:
        return self.view(agent)[0]

    def reward(self, agent: Agent) -> Tensor:
        return torch.zeros(self.world.batch_dim, device=self.world.device)
