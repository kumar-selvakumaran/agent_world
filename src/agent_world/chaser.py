"""Scripted chasers: movable landmarks that hunt agents they can see.

A chaser sees the nearest agent when it is in range, inside its field of view
and not hidden behind a wall or obstacle (one ray, via
:func:`agent_world.raycast.intersect`). If it sees one it turns toward it and
drives at it; otherwise it stops. Touching an agent shoves it along the
chaser-to-agent direction and starts the chaser's cooldown.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from vmas.simulator.core import Landmark, Sphere

from agent_world.patch import inside
from agent_world.raycast import Scene, intersect

NAME = "chaser"
STRIPES = 6
RED = (0.85, 0.10, 0.10)
BLACK = (0.05, 0.05, 0.05)
CANDIDATES = 128
MIN_START_DIST = 4.0
"""Chasers in the agents' room start at least this far from every agent."""


class Chasers:
    def __init__(self, cfg, world, patches):
        self.cfg, self.world, self.patches = cfg, world, patches
        n = cfg.chasers_per_room * cfg.n_rooms
        self.room = [i // cfg.chasers_per_room for i in range(n)]
        self.landmarks = [
            Landmark(
                name=f"{NAME}_{i}", shape=Sphere(cfg.agent_radius), movable=True,
                collide=True, drag=0.0, color=RED,
            )
            for i in range(n)
        ]
        for c in self.landmarks:
            world.add_landmark(c)
        dev = world.device
        self.heading = torch.zeros(world.batch_dim, n, device=dev)
        self.cooldown = torch.zeros(world.batch_dim, n, device=dev)
        self.push_left = None
        self.touched = None

    def _agents(self):
        return [a for a in self.world.agents if a.collide]

    def reset(self, env_index, obstacles, x_bounds, y_bounds):
        """x_bounds(room) -> (lo, hi). Obstacles must already be placed."""
        cfg, dev = self.cfg, self.world.device
        sel = slice(None) if env_index is None else slice(env_index, env_index + 1)
        n = self.world.batch_dim if env_index is None else 1
        agents = torch.stack([a.state.pos[sel] for a in self._agents()], dim=1)
        obst = torch.stack([o.state.pos[sel] for o in obstacles], dim=1) if obstacles else None
        for c, room in zip(self.landmarks, self.room):
            lo, hi = x_bounds(room)
            u = torch.rand(n, CANDIDATES, 2, device=dev)
            cand = torch.stack(
                [lo + u[..., 0] * (hi - lo), y_bounds[0] + u[..., 1] * (y_bounds[1] - y_bounds[0])], -1
            )
            a = torch.cdist(cand, agents).amin(-1) if room == 0 else torch.full_like(cand[..., 0], MIN_START_DIST)
            o = torch.cdist(cand, obst).amin(-1) if obst is not None else torch.full_like(a, 1.0)
            o_min = cfg.obstacle_size / 2 + 2 * cfg.agent_radius
            score = 10 * ((a >= MIN_START_DIST) & (o >= o_min)) + a.clamp(max=MIN_START_DIST) + o.clamp(max=o_min)
            pick = score.argmax(-1)
            c.set_pos(cand[torch.arange(n), pick], batch_index=env_index)
            c.set_vel(torch.zeros(n, 2, device=dev), batch_index=env_index)
            obst = torch.cat([obst, c.state.pos[sel][:, None]], 1) if obst is not None else c.state.pos[sel][:, None]
        self.heading[sel] = (torch.rand(n, len(self.landmarks), device=dev) * 2 - 1) * math.pi
        self.cooldown[sel] = 0.0
        for i, c in enumerate(self.landmarks):
            c.set_rot(self.heading[sel][:, i : i + 1], batch_index=env_index)
        batch = self.world.batch_dim
        if self.push_left is None:
            self.push_left = torch.zeros(batch, len(agents[0]), dtype=torch.long, device=dev)
            self.push_dir = torch.zeros(batch, len(agents[0]), 2, device=dev)
            self.touched = torch.zeros(len(agents[0]), batch, dtype=torch.bool, device=dev)
        self.push_left[sel] = 0
        self.touched[:, sel] = False

    def step(self, occluders: Scene) -> None:
        cfg, world = self.cfg, self.world
        dt = world.dt
        agents = self._agents()
        apos = torch.stack([a.state.pos for a in agents], dim=1)  # [B, A, 2]
        speed = cfg.chaser_speed_fraction * cfg.max_speed
        max_turn = cfg.max_turn_rate * dt
        self.cooldown = (self.cooldown - dt).clamp(min=0)
        self.touched.zero_()
        for i, c in enumerate(self.landmarks):
            pos = c.state.pos
            delta = apos - pos[:, None]
            dist = delta.norm(dim=-1)
            near = dist.argmin(-1)
            rows = torch.arange(pos.shape[0], device=pos.device)
            d, vec = dist[rows, near], delta[rows, near]
            want = torch.atan2(vec[:, 1], vec[:, 0])
            err = torch.remainder(want - self.heading[:, i] + math.pi, 2 * math.pi) - math.pi
            origin = torch.cat([pos, torch.full_like(pos[:, :1], 0.05)], -1)
            ray = torch.cat([vec / d[:, None].clamp(min=1e-6), torch.zeros_like(d)[:, None]], -1)[:, None]
            blocked = intersect(origin, ray, occluders).amin(-1)[:, 0] < d
            sees = (d <= cfg.chaser_view_range) & (err.abs() <= math.radians(cfg.chaser_half_fov_deg)) & ~blocked

            heading = self.heading[:, i] + torch.where(sees, err.clamp(-max_turn, max_turn), 0.0)
            for p in self.patches:
                heading = heading + p.rate * dt * inside(pos, p)
            heading = torch.remainder(heading + math.pi, 2 * math.pi) - math.pi
            self.heading[:, i] = heading
            c.set_rot(heading[:, None], batch_index=None)
            fwd = torch.stack([heading.cos(), heading.sin()], -1)
            c.set_vel(fwd * speed * sees[:, None], batch_index=None)

            for j, a in enumerate(agents):
                hit = world.is_overlapping(c, a) & (self.cooldown[:, i] <= 0)
                direction = delta[:, j] / dist[:, j, None].clamp(min=1e-6)
                self.push_dir[:, j] = torch.where(hit[:, None], direction, self.push_dir[:, j])
                self.push_left[:, j] = torch.where(hit, cfg.bump_steps, self.push_left[:, j])
                self.touched[j] |= hit
                self.cooldown[:, i] = torch.where(hit, cfg.bump_cooldown, self.cooldown[:, i])

        extra = cfg.bump_distance / (cfg.bump_steps * dt)
        for j, a in enumerate(agents):
            active = (self.push_left[:, j] > 0)[:, None]
            a.state.force = a.state.force + active * self.push_dir[:, j] * extra * a.mass / dt
            self.push_left[:, j] = (self.push_left[:, j] - 1).clamp(min=0)
