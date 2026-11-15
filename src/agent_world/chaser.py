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

from agent_world.blocks import NAME as BLOCK, touching
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
                collision_filter=lambda e: not e.name.startswith(BLOCK),
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

    def _spawn(self, room, n, agents, obst, clear):
        """Best of CANDIDATES random spots in ``room``: far from agents (if given), clear of ``obst``."""
        lo, hi = self.x_bounds(room)
        y = self.y_bounds
        u = torch.rand(n, CANDIDATES, 2, device=self.world.device)
        cand = torch.stack([lo + u[..., 0] * (hi - lo), y[0] + u[..., 1] * (y[1] - y[0])], -1)
        a = torch.cdist(cand, agents).amin(-1) if agents is not None else torch.full_like(cand[..., 0], MIN_START_DIST)
        o = (torch.cdist(cand, obst) - clear).amin(-1) if obst is not None else torch.zeros_like(a)
        score = 10 * ((a >= MIN_START_DIST) & (o >= 0)) + a.clamp(max=MIN_START_DIST) + o.clamp(max=0)
        return cand[torch.arange(n), score.argmax(-1)]

    def reset(self, env_index, statics, x_bounds, y_bounds):
        """statics: [(landmarks, clearance)], already placed. x_bounds(room) -> (lo, hi)."""
        cfg, dev = self.cfg, self.world.device
        self.x_bounds, self.y_bounds, self.statics = x_bounds, y_bounds, statics
        sel = slice(None) if env_index is None else slice(env_index, env_index + 1)
        n = self.world.batch_dim if env_index is None else 1
        agents = torch.stack([a.state.pos[sel] for a in self._agents()], dim=1)
        obst, clear = self._static(sel)
        for c, room in zip(self.landmarks, self.room):
            pos = self._spawn(room, n, agents if room == 0 else None, obst, clear)
            c.set_pos(pos, batch_index=env_index)
            c.set_vel(torch.zeros(n, 2, device=dev), batch_index=env_index)
            new = c.state.pos[sel][:, None]
            obst = new if obst is None else torch.cat([obst, new], 1)
            clear = torch.cat([clear, clear.new_full((1,), cfg.obstacle_size / 2 + 2 * cfg.agent_radius)])
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

    def _static(self, sel):
        """Positions [n, K, 2] and clearances [K] of obstacles and blocks (None, None if there are none)."""
        pos = [torch.stack([l.state.pos[sel] for l in ls], 1) for ls, _ in self.statics if ls]
        clear = [torch.full((len(ls),), c, device=self.world.device) for ls, c in self.statics if ls]
        return (torch.cat(pos, 1), torch.cat(clear)) if pos else (None, torch.zeros(0, device=self.world.device))

    def respawn_on_blocks(self, blocks, agents) -> None:
        """Chasers touching a block vanish and reappear in their own room, far from every agent."""
        cfg = self.cfg
        cpos = torch.stack([c.state.pos for c in self.landmarks], dim=1)
        hit = touching(cpos, blocks, cfg.block_size / 2, cfg.agent_radius)  # [B, C]
        obst, clear = self._static(slice(None))
        B = cpos.shape[0]
        for i, (c, room) in enumerate(zip(self.landmarks, self.room)):
            h = hit[:, i, None]
            c.set_pos(torch.where(h, self._spawn(room, B, agents, obst, clear), c.state.pos), batch_index=None)
            c.set_vel(c.state.vel * ~h, batch_index=None)
        self.heading = torch.where(hit, (torch.rand_like(self.heading) * 2 - 1) * math.pi, self.heading)
        self.cooldown = self.cooldown * ~hit

    def step(self, occluders: Scene, blocks=None) -> None:
        cfg, world = self.cfg, self.world
        dt = world.dt
        agents = self._agents()
        apos = torch.stack([a.state.pos for a in agents], dim=1)  # [B, A, 2]
        if blocks is not None:
            self.respawn_on_blocks(blocks, apos)
        cpos = torch.stack([c.state.pos for c in self.landmarks], dim=1)  # [B, C, 2]
        B, C = cpos.shape[:2]
        A = apos.shape[1]
        speed = cfg.chaser_speed_fraction * cfg.max_speed
        max_turn = cfg.max_turn_rate * dt
        self.cooldown = (self.cooldown - dt).clamp(min=0)
        self.touched.zero_()

        delta = apos[:, None] - cpos[:, :, None]  # [B, C, A, 2] chaser to agent
        dist = delta.norm(dim=-1)  # [B, C, A]
        d, near = dist.min(-1)  # nearest agent, [B, C]
        vec = delta.gather(2, near[..., None, None].expand(B, C, 1, 2))[:, :, 0]
        want = torch.atan2(vec[..., 1], vec[..., 0])
        err = torch.remainder(want - self.heading + math.pi, 2 * math.pi) - math.pi
        # One line-of-sight ray per chaser, all cast in a single call.
        origin = torch.cat([cpos, torch.full_like(cpos[..., :1], 0.05)], -1).reshape(B * C, 3)
        ray = torch.cat([vec / d[..., None].clamp(min=1e-6), torch.zeros_like(d)[..., None]], -1).reshape(B * C, 1, 3)
        scene = Scene(
            occluders.pos.repeat_interleave(C, 0), occluders.rot.repeat_interleave(C, 0),
            occluders.half_size, occluders.height, occluders.is_box, occluders.color,
        )
        blocked = intersect(origin, ray, scene).amin(-1).reshape(B, C) < d
        sees = (d <= cfg.chaser_view_range) & (err.abs() <= math.radians(cfg.chaser_half_fov_deg)) & ~blocked

        go = sees & (self.cooldown <= 0)  # after a bump a chaser stands still until its cooldown ends
        heading = self.heading + torch.where(go, err.clamp(-max_turn, max_turn), 0.0)
        for p in self.patches:
            heading = heading + p.rate * dt * inside(cpos, p)
        self.heading = torch.remainder(heading + math.pi, 2 * math.pi) - math.pi
        vel = torch.stack([self.heading.cos(), self.heading.sin()], -1) * speed * go[..., None]
        for i, c in enumerate(self.landmarks):
            c.set_rot(self.heading[:, i : i + 1], batch_index=None)
            c.set_vel(vel[:, i], batch_index=None)

        # Touches: the last chaser to touch an agent decides the direction of the shove.
        # Soft contact settles about 0.004 apart, so allow a little tolerance.
        hit = (dist < 2 * cfg.agent_radius + 0.02) & (self.cooldown[..., None] <= 0)  # [B, C, A]
        any_hit = hit.any(1)  # [B, A]
        last = (hit * torch.arange(C, device=dist.device)[None, :, None]).amax(1)  # [B, A]
        toward = delta.gather(1, last[:, None, :, None].expand(B, 1, A, 2))[:, 0]
        length = dist.gather(1, last[:, None, :])[:, 0, :, None].clamp(min=1e-6)
        self.push_dir = torch.where(any_hit[..., None], toward / length, self.push_dir)
        self.push_left = torch.where(any_hit, cfg.bump_steps, self.push_left)
        self.touched |= any_hit.T
        self.cooldown = torch.where(hit.any(-1), cfg.bump_cooldown, self.cooldown)

        extra = cfg.bump_distance / (cfg.bump_steps * dt)
        for j, a in enumerate(agents):
            active = (self.push_left[:, j] > 0)[:, None]
            a.state.force = a.state.force + active * self.push_dir[:, j] * extra * a.mass / dt
            self.push_left[:, j] = (self.push_left[:, j] - 1).clamp(min=0)
