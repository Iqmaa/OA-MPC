"""Run all five methods on paired random trials, save CSV / summary / plots / GIFs.

    python run_experiments.py --quick                 # ~1 minute smoke test
    python run_experiments.py --trials 40             # full run
    python run_experiments.py --agents 2              # two pedestrians
    python run_experiments.py --no-compensate         # planner ignores the sensor delay (harsher on the baseline)
    ANTHROPIC_API_KEY=... python run_experiments.py --use-llm   # real LLM scene classifier
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import Cfg
from world import World, ZONES
from mpc import MPC
from risk import zone_priors
from sim import run_trial, VARIANTS

LABELS = {"1_baseline": "1 Baseline MPC", "2_oa_mpc": "2 OA-MPC (worst case)", "3_oa+prior": "3 OA + scene prior",
          "4_oa+audio": "4 OA + audio only", "5_oa+prior+audio": "5 OA + prior + audio"}
COLORS = {"1_baseline": "#999999", "2_oa_mpc": "#1f77b4", "3_oa+prior": "#2ca02c",
          "4_oa+audio": "#ff7f0e", "5_oa+prior+audio": "#d62728"}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40, help="trials WITH a hidden agent")
    ap.add_argument("--clean-trials", type=int, default=20, help="trials with NO agent (cost of caution)")
    ap.add_argument("--use-llm", action="store_true", help="use the Anthropic API for scene risk labels")
    ap.add_argument("--agents", type=int, default=1, help="hidden pedestrians per trial (1 or 2)")
    ap.add_argument("--no-compensate", action="store_true", help="planner does NOT correct for the 0.3 s sensor delay")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if a.quick:
        a.trials, a.clean_trials = 8, 4
    os.makedirs(a.out, exist_ok=True)

    cfg = Cfg(n_agents=a.agents, compensate_latency=not a.no_compensate); world = World(cfg); mpc = MPC(cfg)
    priors = {"llm": zone_priors(cfg, "llm", a.use_llm), "uniform": zone_priors(cfg, "uniform")}

    rows, speed_traces = [], {v: [] for v in VARIANTS}
    for with_agent, n in ((True, a.trials), (False, a.clean_trials)):
        for v in VARIANTS:
            for seed in range(n):
                r, (xs, vs), _ = run_trial(v, seed, cfg, world, mpc, priors, with_agent)
                rows.append(r)
                if not with_agent:
                    speed_traces[v].append((xs, vs))
            print(f"done: {'agent' if with_agent else 'clean'} {v}")
    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/trials.csv", index=False)

    # ------------------------------------------------------------- summary table
    out = []
    for v in VARIANTS:
        d1, d0 = df[(df.variant == v) & df.with_agent], df[(df.variant == v) & ~df.with_agent]
        k = int(d1.collided.sum()); lo, hi = wilson(k, len(d1))
        out.append(dict(method=LABELS[v], collisions=f"{k}/{len(d1)}", coll_rate=k / len(d1), ci_lo=lo, ci_hi=hi,
                        near_miss_rate=d1.near_miss.mean(), mean_closest_m=d1.min_dist.mean(), time_with_agent=d1.time.mean(),
                        time_no_agent=d0.time.mean(), v_alcove=d0.v_alcove.mean(), v_junction=d0.v_junction.mean(),
                        solve_ms=df[df.variant == v].solve_ms.mean()))
    S = pd.DataFrame(out)
    S.to_csv(f"{a.out}/summary.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n", S.round(2).to_string(index=False))

    # ------------------------------------------------------------- plot 1: speed vs position
    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.arange(0.5, 13.5, 0.5)
    for v in VARIANTS:
        allx = np.concatenate([x for x, _ in speed_traces[v]]); allv = np.concatenate([s for _, s in speed_traces[v]])
        idx = np.digitize(allx, bins)
        m = [allv[idx == i].mean() if np.any(idx == i) else np.nan for i in range(1, len(bins))]
        ax.plot(bins[:-1] + 0.25, m, label=LABELS[v], color=COLORS[v], lw=2)
    for name, (poly, _) in ZONES.items():
        ax.axvspan(poly.bounds[0], poly.bounds[2], alpha=0.1, color="red" if name == "junction" else "green")
        ax.text((poly.bounds[0] + poly.bounds[2]) / 2, 0.05, name, ha="center", fontsize=8)
    ax.set_xlabel("position along corridor (m)"); ax.set_ylabel("mean speed (m/s)")
    ax.set_title("Speed profile when NO agent is present: who slows down where?"); ax.legend(fontsize=7); ax.set_ylim(0, 1.8)
    fig.tight_layout(); fig.savefig(f"{a.out}/speed_profile.png", dpi=140); plt.close(fig)

    # ------------------------------------------------------------- plot 2: safety vs speed
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    names = [LABELS[v] for v in VARIANTS]; cols = [COLORS[v] for v in VARIANTS]; xi = np.arange(len(names))
    err = np.array([[r.coll_rate - r.ci_lo, r.ci_hi - r.coll_rate] for r in S.itertuples()]).T
    axs[0].bar(xi, S.coll_rate * 100, yerr=err * 100, color=cols, capsize=3); axs[0].set_title("Collision rate (%) - agent present\n(95% Wilson CI)")
    axs[1].bar(xi, S.near_miss_rate * 100, color=cols); axs[1].set_title("Near-miss rate (%) - agent present")
    axs[2].bar(xi, S.time_no_agent, color=cols); axs[2].set_title("Time to goal (s) - no agent\n(cost of being cautious)")
    for ax in axs:
        ax.set_xticks(xi); ax.set_xticklabels([n.split(" ", 1)[0] for n in names])
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in cols], names, loc="lower center", ncol=5, fontsize=7)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(f"{a.out}/summary.png", dpi=140); plt.close(fig)

    print(f"\nSaved results to ./{a.out}/")


if __name__ == "__main__":
    main()
