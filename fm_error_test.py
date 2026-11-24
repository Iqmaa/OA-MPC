"""What if the foundation model gets the scene WRONG?  (safety check for the risk prior)

Runs method 3 (OA + prior) with the junction / alcove labels set correctly and incorrectly.
    python fm_error_test.py --trials 60
"""
import argparse
import numpy as np
import pandas as pd
from config import Cfg
from world import World, ZONE_NAMES
from mpc import MPC
from sim import run_trial

ap = argparse.ArgumentParser()
ap.add_argument("--trials", type=int, default=60)
ap.add_argument("--agents", type=int, default=1)
a = ap.parse_args()
cfg = Cfg(n_agents=a.agents); world = World(cfg); mpc = MPC(cfg)
pm = dict(cfg.prior_map)
CASES = {   # (alcove label, junction label)
    "correct   (alcove low,  junction high)":   ("low", "high"),
    "FM wrong  (junction labelled LOW)":        ("low", "low"),
    "FM wrong  (junction labelled MEDIUM)":     ("low", "medium"),
    "FM cautious (alcove labelled HIGH)":       ("high", "high"),
}
rows = []
for name, (al, jn) in CASES.items():
    pri = np.array([pm[al] if z == "alcove" else pm[jn] for z in ZONE_NAMES])
    R = [run_trial("3_oa+prior", s, cfg, world, mpc, {"llm": pri, "uniform": pri}, True)[0] for s in range(a.trials)]
    C = [run_trial("3_oa+prior", s, cfg, world, mpc, {"llm": pri, "uniform": pri}, False)[0] for s in range(20)]
    rows.append(dict(case=name, collisions=f"{sum(r['collided'] for r in R)}/{a.trials}",
                     closest_m=np.mean([r["min_dist"] for r in R]), time_no_agent=np.nanmean([r["time"] for r in C])))
    print(rows[-1])
pd.DataFrame(rows).to_csv("results/fm_error_test.csv", index=False)
