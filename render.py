"""Video rendering for recorded trials (MP4 via ffmpeg, GIF fallback)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Circle
from world import ZONES

SUB = 3      # sub-frames per control step (dt = 0.2 s -> 15 fps is real time)
FPS = 15
HOLD = 20    # frames to hold the final state


def _writer(path):
    if path.endswith(".mp4"):
        try:
            import imageio_ffmpeg
            matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
        if FFMpegWriter.isAvailable():
            return FFMpegWriter(fps=FPS, bitrate=2500)
        print("ffmpeg not found -> writing a GIF instead (pip install imageio-ffmpeg for MP4)")
        return None
    return PillowWriter(fps=FPS)


class Panel:
    def __init__(self, ax, world, cfg, tr, res, title, color):
        self.w, self.cfg, self.tr, self.res = world, cfg, tr, res
        self.n = len(tr["t"])
        ax.set_aspect("equal"); ax.set_xlim(-0.3, 14.3); ax.set_ylim(-0.3, 9.6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11, color=color, fontweight="bold")
        # what the robot can see: precompute one image per control step
        self.imgs = []
        for i in range(self.n):
            P = np.array([tr["x"][i], tr["y"][i]])
            d = np.linalg.norm(world.cells - P, axis=1)
            vis = (~world.blocked(P, world.cells)) & (d <= cfg.sensor_range)
            v2 = np.zeros(world.free2.shape, bool); v2[world.ci, world.cj] = vis
            im = np.zeros(world.free2.shape + (4,))
            im[world.free2 & ~v2] = (0.15, 0.15, 0.2, 0.60)       # hidden from the robot
            im[world.free2 & v2] = (1.0, 0.96, 0.65, 0.35)        # currently visible
            self.imgs.append(im)
        self.im = ax.imshow(self.imgs[0], extent=(0, 14, 0, 9.5), origin="lower", zorder=0, interpolation="nearest")
        for a, b in world.walls:
            ax.plot([a[0], b[0]], [a[1], b[1]], "k-", lw=2, zorder=3)
        for name, (poly, _) in ZONES.items():
            ax.text(poly.centroid.x, poly.bounds[3] + 0.15, name, ha="center", fontsize=8,
                    color="darkred" if name == "junction" else "darkgreen")
        ax.plot([cfg.goal_x] * 2, [0, 2.5], "g--", lw=1.5, zorder=3)
        self.discs = [Circle((1e3, 1e3), 0.1, fc="orange", ec="darkorange", alpha=0.15, zorder=2) for _ in range(cfg.max_discs)]
        for c in self.discs:
            ax.add_patch(c)
        self.robot = Circle((0, 0), cfg.r_robot, fc="tab:blue", ec="navy", zorder=5); ax.add_patch(self.robot)
        self.agents = [Circle((-9, -9), cfg.r_agent, fc="red", ec="darkred", zorder=4) for _ in tr["ag"][0]]
        for c in self.agents:
            ax.add_patch(c)
        self.txt = ax.text(0.15, 9.35, "", fontsize=9, va="top", color="white",
                           bbox=dict(fc="black", alpha=0.6, pad=2), zorder=6)
        self.banner = ax.text(7, 6.0, "", fontsize=20, ha="center", fontweight="bold", zorder=7)
        self.nframes = self.n * SUB

    def update(self, f):
        tr = self.tr
        i = min(f // SUB, self.n - 1)
        j = min(i + 1, self.n - 1)
        a = (f % SUB) / SUB if i < self.n - 1 else 0.0
        lerp = lambda u, v: u + a * (v - u)
        self.robot.center = (lerp(tr["x"][i], tr["x"][j]), lerp(tr["y"][i], tr["y"][j]))
        self.im.set_data(self.imgs[i])
        for k, c in enumerate(self.agents):
            p, vis = tr["ag"][i][k]
            if p is None:
                c.center = (-9, -9); continue
            q = tr["ag"][j][k][0]
            pos = p if q is None else (lerp(p[0], q[0]), lerp(p[1], q[1]))
            c.center = tuple(pos)
            c.set_alpha(0.95 if vis else 0.35)
            c.set_linestyle("-" if vis else "--")
        fr, rad = tr["discs"][i], tr["rad"][i]
        for m, c in enumerate(self.discs):
            if m < len(fr):
                c.center, c.set_radius = tuple(fr[m]), c.set_radius
                c.set_radius(float(rad[m])); c.set_visible(True)
            else:
                c.set_visible(False)
        self.txt.set_text(f"t={tr['t'][i]:.1f}s  speed={tr['v'][i]:.2f} m/s  belief={tr['p'][i]:.2f}")
        if f >= self.nframes - 1:
            if self.res["collided"]:
                self.banner.set_text("COLLISION"); self.banner.set_color("red")
            elif self.res["reached"]:
                self.banner.set_text("GOAL"); self.banner.set_color("green")
        else:
            self.banner.set_text("")


def render(panels_spec, path, suptitle=None, ncols=1):
    """panels_spec: list of (title, color, trace, result, world, cfg)."""
    n = len(panels_spec)
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(6.6 * ncols, 4.55 * nrows + (0.5 if suptitle else 0)), squeeze=False)
    panels = [Panel(axs[k // ncols][k % ncols], w, c, tr, res, title, col)
              for k, (title, col, tr, res, w, c) in enumerate(panels_spec)]
    for k in range(n, nrows * ncols):
        ax = axs[k // ncols][k % ncols]; ax.axis("off")
        ax.text(0.02, 0.98, "How to read this\n\n"
                "dark area      = hidden from the robot\n"
                "yellow area    = currently visible\n"
                "orange circles = where an unseen person could\n"
                "                 be within the planning horizon\n"
                "red circle     = pedestrian (faded/dashed =\n"
                "                 the robot cannot see them)\n"
                "blue circle    = robot, green line = goal\n"
                "belief         = P(someone about to emerge)\n"
                "                 in the nearest hidden zone",
                va="top", family="monospace", fontsize=9, transform=ax.transAxes)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    total = max(p.nframes for p in panels) + HOLD
    for p in panels:                       # shorter trials freeze on their last frame
        p_nf = p.nframes
        p.update_orig = p.update
        p.update = (lambda f, p=p, nf=p_nf: p.update_orig(min(f, nf - 1)))
    anim = FuncAnimation(fig, lambda f: [p.update(f) for p in panels], frames=total, blit=False)
    wr = _writer(path)
    if wr is None:
        path = path.replace(".mp4", ".gif"); wr = PillowWriter(fps=FPS)
    anim.save(path, writer=wr, dpi=80)
    plt.close(fig)
    return path
