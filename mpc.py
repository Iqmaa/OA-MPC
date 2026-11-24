"""Nonlinear MPC (CasADi + IPOPT) with OA-MPC-style occlusion discs.

State  x = [px, py, heading, speed]     Control u = [acceleration, turn rate]
All obstacles (walls, occlusion discs, visible agent) are SOFT penalties, so the problem is always
feasible. (The original OA-MPC uses hard constraints and proves recursive feasibility; see README.)

The problem is built ONCE and re-solved every step with new numeric parameters.
"""
import casadi as ca
import numpy as np


class MPC:
    def __init__(self, cfg):
        self.cfg = cfg
        N, M, W, dt = cfg.N, cfg.max_discs, cfg.wall_pts, cfg.dt
        K = cfg.max_agents
        X = ca.SX.sym("X", 4, N + 1)
        U = ca.SX.sym("U", 2, N)
        x0 = ca.SX.sym("x0", 4)
        ref = ca.SX.sym("ref", 2, N)
        vref = ca.SX.sym("vref")
        wall = ca.SX.sym("wall", W, 2)
        disc = ca.SX.sym("disc", M, 2)
        rad = ca.SX.sym("rad", M, N)
        ag = ca.SX.sym("ag", K * N, 2)
        agr = ca.SX.sym("agr", K * N)
        wstop = ca.SX.sym("wstop")
        P = ca.vertcat(x0, ca.vec(ref), vref, ca.vec(wall), ca.vec(disc), ca.vec(rad),
                       ca.vec(ag), agr, wstop)

        g = [X[:, 0] - x0]
        J = 0
        r_w = cfg.r_robot + 0.08
        for k in range(N):
            x, u, xn = X[:, k], U[:, k], X[:, k + 1]
            g.append(xn - ca.vertcat(x[0] + x[3] * ca.cos(x[2]) * dt,
                                     x[1] + x[3] * ca.sin(x[2]) * dt,
                                     x[2] + u[1] * dt,
                                     x[3] + u[0] * dt))
            p = xn[:2]
            J += cfg.w_pos * ca.sumsqr(p - ref[:, k]) + cfg.w_v * (xn[3] - vref) ** 2
            J += cfg.w_acc * u[0] ** 2 + cfg.w_turn * u[1] ** 2 + cfg.w_yaw * xn[2] ** 2
            for w in range(W):                      # walls
                d = ca.sqrt(ca.sumsqr(p - wall[w, :].T) + 1e-6)
                J += cfg.w_wall * ca.fmax(0, r_w - d) ** 2
            for m in range(M):                      # occlusion discs (OA-MPC reachable sets)
                d = ca.sqrt(ca.sumsqr(p - disc[m, :].T) + 1e-6)
                J += cfg.w_disc * ca.fmax(0, rad[m, k] + cfg.disc_margin - d) ** 2
            for j in range(K):                      # visible agents (predicted positions)
                d = ca.sqrt(ca.sumsqr(p - ag[j * N + k, :].T) + 1e-6)
                J += cfg.w_agent * ca.fmax(0, agr[j * N + k] - d) ** 2
        J += wstop * X[3, N] ** 2                   # soft "be able to stop at the end of the horizon"

        z = ca.vertcat(ca.vec(X), ca.vec(U))
        opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes", "ipopt.max_iter": 80,
                "ipopt.tol": 1e-3, "ipopt.acceptable_tol": 1e-2,
                "ipopt.warm_start_init_point": "yes", "ipopt.mu_init": 1e-2}
        self.solver = ca.nlpsol("mpc", "ipopt", {"x": z, "f": J, "g": ca.vertcat(*g), "p": P}, opts)
        nx = 4 * (N + 1)
        self.lbx = np.concatenate([np.tile([-1e3, -1e3, -3.0, 0.0], N + 1),
                                   np.tile([-cfg.a_max, -cfg.w_max], N)])
        self.ubx = np.concatenate([np.tile([1e3, 1e3, 3.0, cfg.v_max], N + 1),
                                   np.tile([cfg.a_max, cfg.w_max], N)])
        self.ng = 4 * (N + 1)
        self.nx = nx
        self.guess = None

    def reset(self):
        self.guess = None

    def solve(self, x0, ref, vref, wall, disc, rad, ag, agr, wstop):
        cfg = self.cfg
        N, M, W = cfg.N, cfg.max_discs, cfg.wall_pts
        wall_p = np.full((W, 2), 1e3)
        wall_p[: len(wall)] = wall[:W]
        disc_p = np.full((M, 2), 1e3)
        rad_p = np.zeros((M, N))
        if len(disc):
            disc_p[: len(disc)] = disc[:M]
            rad_p[: len(disc)] = rad[:M]
        p = np.concatenate([x0, ref.ravel(order="F"), [vref], wall_p.ravel(order="F"),
                            disc_p.ravel(order="F"), rad_p.ravel(order="F"),
                            ag.ravel(order="F"), agr, [wstop]])
        if self.guess is None:
            Xg = np.array([x0 + np.array([0, 0, 0, 0]) for _ in range(N + 1)]).T
            for k in range(1, N + 1):
                Xg[0, k] = x0[0] + x0[3] * cfg.dt * k
            self.guess = np.concatenate([Xg.ravel(order="F"), np.zeros(2 * N)])
        sol = self.solver(x0=self.guess, p=p, lbx=self.lbx, ubx=self.ubx, lbg=0, ubg=0)
        ok = self.solver.stats()["success"] or self.solver.stats()["return_status"] in (
            "Solved_To_Acceptable_Level", "Maximum_Iterations_Exceeded")
        z = np.array(sol["x"]).ravel()
        Xs = z[: self.nx].reshape(N + 1, 4)
        Us = z[self.nx:].reshape(N, 2)
        if not ok or not np.all(np.isfinite(z)):
            self.guess = None
            return np.array([-cfg.a_max, 0.0]), None
        # warm start: shift the plan forward by one step
        Xn = np.vstack([Xs[1:], Xs[-1:]])
        Un = np.vstack([Us[1:], Us[-1:]])
        self.guess = np.concatenate([Xn.ravel(), Un.ravel()])
        return Us[0], Xs
