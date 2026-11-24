"""2D world: a corridor with a harmless alcove (low risk) and a blind side-corridor crossing (high risk).

            y=9  +--+
                 |  |   <- side corridor ("junction" zone), x in [6, 8.5], hidden from the robot
                 |  |      until it is almost level with it (T-junction: it only goes north)
      +---+      |  |
      |alc|      |  |
 y=2.5+-+ +------+  +------------+
      robot -->  .  .  .  .  .  .   goal         (main corridor, y in [0, 2.5])
 y=0  +----------------------------+

The pedestrian walks down the side corridor, crosses the main corridor and leaves through a door in the
south wall (it simply disappears at y = 0.4).
"""
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union
from shapely import contains_xy

# name -> (shapely polygon, plain-text description given to the scene classifier)
ZONES = {
    "alcove": (
        box(2.0, 2.5, 3.0, 3.5),
        "A shallow storage alcove set into the corridor wall, one metre deep. It has no door "
        "and no other exit, so nothing can enter it from anywhere else.",
    ),
    "junction": (
        box(6.0, 0.0, 8.5, 9.0),
        "A blind crossing where a busy side corridor meets the robot's corridor at a right angle. "
        "A wall hides the side corridor until the robot reaches the junction, and people "
        "regularly walk through it.",
    ),
}
ZONE_NAMES = list(ZONES.keys())


class World:
    def __init__(self, cfg):
        self.cfg = cfg
        rects = [box(0, 0, 14, 2.5), box(6, 0, 8.5, 9), box(2, 2.5, 3, 3.5)]
        self.free = unary_union(rects)
        c = np.array(self.free.exterior.coords)
        self.walls = np.stack([c[:-1], c[1:]], axis=1)  # (S,2,2) wall segments

        # sample points along the walls (used as soft collision points in the MPC)
        pts = []
        for a, b in self.walls:
            n = max(2, int(np.linalg.norm(b - a) / 0.2) + 1)
            for s in np.linspace(0, 1, n):
                pts.append(a + s * (b - a))
        self.wall_pts = np.unique(np.round(np.array(pts), 3), axis=0)

        # visibility grid over free space
        res = cfg.grid_res
        xs = np.arange(res / 2, 14, res)
        ys = np.arange(res / 2, 9.5, res)
        self.X, self.Y = np.meshgrid(xs, ys)            # shape (ny, nx)
        self.free2 = contains_xy(self.free, self.X, self.Y)
        self.ci, self.cj = np.nonzero(self.free2)
        self.cells = np.stack([self.X[self.ci, self.cj], self.Y[self.ci, self.cj]], axis=1)
        self.cell_zone = -np.ones(len(self.cells), dtype=int)
        for zi, name in enumerate(ZONE_NAMES):
            m = contains_xy(ZONES[name][0], self.cells[:, 0], self.cells[:, 1])
            self.cell_zone[m] = zi

    # ------------------------------------------------------------------ visibility
    def blocked(self, P, Q):
        """True where the segment P->Q[i] crosses a wall. P:(2,), Q:(C,2)."""
        A, B = self.walls[:, 0, :], self.walls[:, 1, :]
        d1, d2, AP = Q - P, B - A, A - P
        den = d1[:, None, 0] * d2[None, :, 1] - d1[:, None, 1] * d2[None, :, 0]
        ok = np.abs(den) > 1e-12
        den = np.where(ok, den, 1.0)
        t = (AP[None, :, 0] * d2[None, :, 1] - AP[None, :, 1] * d2[None, :, 0]) / den
        u = (AP[None, :, 0] * d1[:, None, 1] - AP[None, :, 1] * d1[:, None, 0]) / den
        e = 1e-9
        return (ok & (t > e) & (t < 1 - e) & (u > e) & (u < 1 - e)).any(axis=1)

    def zone_distance(self, P):
        from shapely.geometry import Point
        pt = Point(float(P[0]), float(P[1]))
        return np.array([ZONES[n][0].distance(pt) for n in ZONE_NAMES])

    def can_see(self, P, q):
        P = np.asarray(P, float)
        q = np.asarray(q, float)
        if np.linalg.norm(q - P) > self.cfg.sensor_range:
            return False
        return not bool(self.blocked(P, q[None, :])[0])

    def perceive(self, P):
        """Return (frontier_points, frontier_zone_ids, occluded_zone_mask, ahead_cells_by_zone).

        frontier = occluded free cells that touch a visible free cell (where an unseen agent could emerge).
        """
        cfg = self.cfg
        P = np.asarray(P, float)
        d = np.linalg.norm(self.cells - P, axis=1)
        vis = (~self.blocked(P, self.cells)) & (d <= cfg.sensor_range)
        vis2 = np.zeros(self.free2.shape, bool)
        vis2[self.ci, self.cj] = vis
        occ2 = self.free2 & ~vis2
        nb = np.zeros_like(vis2)
        nb[1:, :] |= vis2[:-1, :]
        nb[:-1, :] |= vis2[1:, :]
        nb[:, 1:] |= vis2[:, :-1]
        nb[:, :-1] |= vis2[:, 1:]
        fr2 = occ2 & nb
        fr_cells = np.stack([self.X[fr2], self.Y[fr2]], axis=1)

        # zone bookkeeping: which zones still contain unseen cells ahead of the robot?
        occ = ~vis & (d <= cfg.track_range) & (self.cells[:, 0] >= P[0] - cfg.fwd_margin)
        zone_occluded = np.zeros(len(ZONE_NAMES), bool)
        for zi, name in enumerate(ZONE_NAMES):
            xmax = ZONES[name][0].bounds[2]
            passed = P[0] > xmax - 0.2
            zone_occluded[zi] = (not passed) and bool(np.any(occ & (self.cell_zone == zi)))

        # frontier -> discs (nearest first, thinned)
        if len(fr_cells):
            dd = np.linalg.norm(fr_cells - P, axis=1)
            keep = (dd <= cfg.frontier_range) & (fr_cells[:, 0] >= P[0] - cfg.fwd_margin)
            fr_cells, dd = fr_cells[keep], dd[keep]
            order = np.argsort(dd)
            chosen = []
            for i in order:
                if all(np.linalg.norm(fr_cells[i] - fr_cells[j]) >= cfg.disc_spacing for j in chosen):
                    chosen.append(i)
                if len(chosen) >= cfg.max_discs:
                    break
            fr_cells = fr_cells[chosen]
        zone_ids = np.full(len(fr_cells), -1, int)
        for zi, name in enumerate(ZONE_NAMES):
            if len(fr_cells):
                zone_ids[contains_xy(ZONES[name][0], fr_cells[:, 0], fr_cells[:, 1])] = zi
        return fr_cells, zone_ids, zone_occluded


class Agent:
    """Hidden pedestrian walking straight down the side corridor, crossing y=1.25 at time t_cross."""

    def __init__(self, x, speed, t_cross, cfg):
        self.x, self.v, self.t_cross, self.cfg = x, speed, t_cross, cfg
        self.y_exit = 0.4          # leaves through a door in the south wall

    def pos(self, t):
        return np.array([self.x, 1.25 + self.v * (self.t_cross - t)])

    def active(self, t):
        return self.pos(t)[1] > self.y_exit

    def moving(self, t):
        return self.active(t)


def sample_agent(seed, cfg, with_agent=True):
    """Same seed -> same agent for every method (paired comparison)."""
    if not with_agent:
        return None
    rng = np.random.default_rng(seed)
    x = rng.uniform(6.45, 8.05)
    v = rng.uniform(*cfg.agent_speed_range)
    t_acc = cfg.v_max / cfg.a_max                       # time to accelerate from rest
    d_acc = 0.5 * cfg.v_max * t_acc
    t_arrive = t_acc + (x - cfg.start[0] - d_acc) / cfg.v_max   # when a naive full-speed robot gets there
    return Agent(x, v, t_arrive + rng.uniform(-0.7, 0.7), cfg)


def sample_agents(seed, cfg, n):
    """n hidden pedestrians. Agent 0 is exactly the one sample_agent() gives (so 1-agent results are
    unchanged). Extra agents follow through the same side corridor 1.2 to 2.5 s later, like a small stream."""
    if n <= 0:
        return []
    agents = [sample_agent(seed, cfg, True)]
    rng = np.random.default_rng(seed + 777)
    for _ in range(n - 1):
        x = rng.uniform(6.45, 8.05)
        v = rng.uniform(*cfg.agent_speed_range)
        agents.append(Agent(x, v, agents[-1].t_cross + rng.uniform(1.2, 2.5), cfg))
    return agents
