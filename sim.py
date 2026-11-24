"""Closed-loop simulation of one trial for one method."""
import time
import numpy as np
from world import World, ZONE_NAMES, sample_agents
from mpc import MPC
from risk import Belief, simulate_audio

# name -> dict(oa: use occlusion discs, prior: None|'llm'|'uniform', audio: bool)
VARIANTS = {
    "1_baseline":      dict(oa=False, prior=None,      audio=False),
    "2_oa_mpc":        dict(oa=True,  prior=None,      audio=False),
    "3_oa+prior":      dict(oa=True,  prior="llm",     audio=False),
    "4_oa+audio":      dict(oa=True,  prior="uniform", audio=True),
    "5_oa+prior+audio": dict(oa=True, prior="llm",     audio=True),
}


def run_trial(variant, seed, cfg, world, mpc, priors, with_agent=True, record=False):
    spec = VARIANTS[variant]
    agents = sample_agents(seed, cfg, cfg.n_agents if with_agent else 0)
    K = cfg.max_agents
    rng_audio = np.random.default_rng(seed + 10_000)
    belief = Belief(cfg, priors[spec["prior"]], spec["audio"]) if spec["prior"] else None
    mpc.reset()

    s = np.array([cfg.start[0], cfg.start[1], 0.0, 0.0])
    t, collided, min_d = 0.0, False, np.inf
    prev_seen = {}
    trace = dict(t=[], x=[], y=[], v=[], p=[], ag=[], discs=[], rad=[]) if record else None
    xs, vs = [], []
    solve_time = []
    N, dt = cfg.N, cfg.dt

    while t < cfg.t_max and s[0] < cfg.goal_x:
        P = s[:2]
        # ---- perception
        vis_flags = [a.active(t) and world.can_see(P, a.pos(t)) for a in agents]
        fr, zid, zone_occ = world.perceive(P) if spec["oa"] else (np.zeros((0, 2)), np.zeros(0, int), np.zeros(len(ZONE_NAMES), bool))

        # ---- belief / risk layer
        heard = None
        if belief is not None:
            if spec["audio"]:
                heard = simulate_audio(cfg, rng_audio, agents, P, t, vis_flags)
            belief.update(zone_occ, heard)

        # ---- occlusion discs and their growth rates
        if len(fr):
            if belief is None:
                scale = np.ones(len(fr))
            else:
                scale = np.clip(belief.p[zid] / cfg.p_full, 0, 1)
                scale[zid < 0] = 1.0
            rad = cfg.r_robot + cfg.r_agent + scale[:, None] * cfg.v_agent_max * dt * np.arange(1, N + 1)[None, :]
        else:
            rad = np.zeros((0, N))

        # ---- speed target and stop requirement
        if belief is None:
            vref = cfg.v_max
            wstop = cfg.w_stop if (spec["oa"] and len(fr)) else 0.0   # only needed while something is hidden
        else:
            near = zone_occ & (world.zone_distance(P) <= cfg.slow_range)
            p_act = belief.p[near].max() if near.any() else 0.0
            vref = cfg.v_min + (cfg.v_max - cfg.v_min) * (1 - np.clip(p_act / cfg.p_full, 0, 1))
            wstop = cfg.w_stop if p_act >= cfg.p_stop else 0.0

        # ---- visible agents: constant-velocity prediction (worst case if velocity not yet known)
        ag = np.full((K * N, 2), 1e3)
        agr = np.zeros(K * N)
        ks = np.arange(1, N + 1)[:, None] * dt
        new_prev = {}
        for i, a in enumerate(agents):
            if not vis_flags[i] or i >= K:
                continue
            pos = a.pos(t - cfg.latency)              # delayed measurement
            sl = slice(i * N, (i + 1) * N)
            lag = cfg.latency if cfg.compensate_latency else 0.0     # extrapolate over the known sensor delay
            if i in prev_seen:
                vel = (pos - prev_seen[i]) / dt
                ag[sl] = pos[None, :] + vel[None, :] * (ks + lag)
                agr[sl] = cfg.r_robot + cfg.r_agent + 0.1
            else:                                                     # first sighting: velocity unknown
                ag[sl] = np.tile(pos, (N, 1))
                agr[sl] = cfg.r_robot + cfg.r_agent + cfg.v_agent_max * (ks[:, 0] + lag)
            new_prev[i] = pos
        prev_seen = new_prev

        # ---- plan
        wd = np.linalg.norm(world.wall_pts - P, axis=1)
        wall = world.wall_pts[np.argsort(wd)[: cfg.wall_pts]]
        ref = np.stack([np.minimum(s[0] + vref * dt * np.arange(1, N + 1), cfg.goal_x + 1),
                        np.full(N, cfg.start[1])])
        t0 = time.perf_counter()
        u, plan = mpc.solve(s, ref, vref, wall, fr, rad, ag, agr, wstop)
        solve_time.append(time.perf_counter() - t0)

        if record:
            trace["t"].append(t); trace["x"].append(s[0]); trace["y"].append(s[1]); trace["v"].append(s[3])
            trace["p"].append(0.0 if belief is None else float(p_act))
            trace["ag"].append([(a.pos(t).copy() if a.active(t) else None, bool(f)) for a, f in zip(agents, vis_flags)])
            trace["discs"].append(fr.copy()); trace["rad"].append(rad[:, -1].copy() if len(rad) else np.zeros(0))
        xs.append(s[0]); vs.append(s[3])

        # ---- apply control (same discrete model as the planner)
        s = np.array([s[0] + s[3] * np.cos(s[2]) * dt, s[1] + s[3] * np.sin(s[2]) * dt,
                      s[2] + u[1] * dt, np.clip(s[3] + u[0] * dt, 0, cfg.v_max)])
        t += dt
        for a in agents:
            if a.active(t):
                d = np.linalg.norm(a.pos(t) - s[:2])
                min_d = min(min_d, d)
                if d < cfg.r_robot + cfg.r_agent:
                    collided = True
        if collided:
            break

    xs, vs = np.array(xs), np.array(vs)
    reached = s[0] >= cfg.goal_x and not collided
    band = lambda lo, hi: float(vs[(xs >= lo) & (xs <= hi)].mean()) if np.any((xs >= lo) & (xs <= hi)) else np.nan
    near = bool(np.isfinite(min_d) and min_d < cfg.r_robot + cfg.r_agent + cfg.near_miss)
    res = dict(variant=variant, seed=seed, with_agent=with_agent, collided=collided, near_miss=near, reached=reached,
               time=t if reached else np.nan, min_dist=min_d if np.isfinite(min_d) else np.nan,
               v_alcove=band(1.0, 3.0), v_junction=band(4.0, 6.5), mean_v=float(vs.mean()),
               solve_ms=1000 * float(np.mean(solve_time)))
    return res, (xs, vs), trace
