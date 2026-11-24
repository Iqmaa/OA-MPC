"""Render videos of the five methods on the SAME trial.

    python make_videos.py                      # auto-picks a telling seed (baseline collides, OA-MPC does not)
    python make_videos.py --agents 2 --seed 7  # two pedestrians, chosen seed
    python make_videos.py --no-compensate      # harsher perception delay
"""
import argparse
import os
from config import Cfg
from world import World
from mpc import MPC
from risk import zone_priors
from sim import run_trial, VARIANTS
from render import render

TITLES = {"1_baseline": "1  Baseline MPC (sees only what is visible)",
          "2_oa_mpc": "2  OA-MPC (worst case for every hidden area)",
          "3_oa+prior": "3  OA + scene-risk prior",
          "4_oa+audio": "4  OA + audio cue only",
          "5_oa+prior+audio": "5  OA + prior + audio"}
COLORS = {"1_baseline": "dimgray", "2_oa_mpc": "tab:blue", "3_oa+prior": "tab:green",
          "4_oa+audio": "darkorange", "5_oa+prior+audio": "tab:red"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-compensate", action="store_true")
    ap.add_argument("--use-llm", action="store_true")
    ap.add_argument("--out", default="videos")
    ap.add_argument("--no-individual", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = Cfg(n_agents=a.agents, compensate_latency=not a.no_compensate)
    world, mpc = World(cfg), MPC(cfg)
    priors = {"llm": zone_priors(cfg, "llm", a.use_llm), "uniform": zone_priors(cfg, "uniform")}

    seed = a.seed
    if seed is None:      # first seed where the baseline collides but OA-MPC (2) and OA+prior (3) do not
        for s in range(200):
            if run_trial("1_baseline", s, cfg, world, mpc, priors, True)[0]["collided"] and \
               not run_trial("2_oa_mpc", s, cfg, world, mpc, priors, True)[0]["collided"] and \
               not run_trial("3_oa+prior", s, cfg, world, mpc, priors, True)[0]["collided"]:
                seed = s
                break
        if seed is None:
            seed = 0
            print("no seed found where baseline collides and 2,3 do not; using seed 0")
    print(f"Using seed {seed}, agents={a.agents}, delay compensated={not a.no_compensate}")

    spec = []
    for v in VARIANTS:
        res, _, tr = run_trial(v, seed, cfg, world, mpc, priors, True, record=True)
        spec.append((TITLES[v], COLORS[v], tr, res, world, cfg))
        print(f"  {v:18s} collided={res['collided']}  closest={res['min_dist']:.2f} m  time={res['time']:.1f}")
        if not a.no_individual:
            render([spec[-1]], f"{a.out}/{v}_seed{seed}.mp4")
    tag = f"{a.agents}agent" + ("s" if a.agents > 1 else "")
    p = render(spec, f"{a.out}/compare_{tag}_seed{seed}.mp4", ncols=2,
               suptitle=f"Same trial, five methods ({a.agents} hidden pedestrian{'s' if a.agents > 1 else ''}, seed {seed} - "
                        f"chosen because the baseline collides)" if a.seed is None else None)
    print("saved", p)


if __name__ == "__main__":
    main()
