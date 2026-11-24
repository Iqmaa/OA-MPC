"""All tunable parameters in one place."""
from dataclasses import dataclass


@dataclass
class Cfg:
    # --- timing / horizon (OA-MPC used 10 Hz, N=5; we use a slightly longer horizon) ---
    dt: float = 0.2
    N: int = 5

    # --- robot (unicycle with speed state) ---
    r_robot: float = 0.3
    v_max: float = 1.6
    v_min: float = 0.9          # slowest "cruise" speed used by the risk layer
    a_max: float = 1.0
    w_max: float = 2.0
    start: tuple = (0.6, 1.25)
    goal_x: float = 13.0
    t_max: float = 40.0

    # --- hidden agent (ground truth in the sim) ---
    r_agent: float = 0.3
    agent_speed_range: tuple = (0.9, 1.4)
    n_agents: int = 1             # hidden pedestrians per trial (the MPC has slots for max_agents)
    max_agents: int = 2
    # --- what the PLANNER assumes about hidden agents (worst case) ---
    v_agent_max: float = 1.4

    # --- perception ---
    sensor_range: float = 12.0
    latency: float = 0.3          # pedestrians are seen where they were this many seconds ago
    compensate_latency: bool = True   # planner extrapolates visible agents over the known delay
    near_miss: float = 0.3        # a 'near miss' = closest approach within this much of actual contact
    grid_res: float = 0.25
    frontier_range: float = 5.0   # only frontier points this close matter for planning
    track_range: float = 9.0      # zones this close are tracked by the belief
    slow_range: float = 2.5       # a zone only changes speed/stop behaviour when the robot is this close
    fwd_margin: float = 0.3       # ignore frontier cells more than this far behind the robot
    max_discs: int = 14
    disc_spacing: float = 0.45
    wall_pts: int = 60

    # --- MPC cost weights ---
    w_pos: float = 6.0
    w_v: float = 1.0
    w_acc: float = 0.05
    w_turn: float = 0.1
    w_yaw: float = 2.0
    w_wall: float = 300.0
    w_disc: float = 300.0
    w_agent: float = 300.0
    w_stop: float = 25.0          # penalty on terminal speed (soft "come to a stop" constraint)
    disc_margin: float = 0.05

    # --- risk layer ---
    prior_map: tuple = (("low", 0.03), ("medium", 0.20), ("high", 0.50))
    p_full: float = 0.5           # belief at which discs grow at the full worst-case rate
    p_stop: float = 0.15          # belief above which the terminal-stop requirement is switched on
    p_clear: float = 0.01         # belief in a zone once we have SEEN all of it (peek)
    p_floor: float = 0.01
    silence_floor: float = 0.4    # silence alone can never push belief below this fraction of the prior
    leak: float = 0.02            # per-step pull of the belief back toward its prior
    uniform_prior: float = 0.20   # prior used when no scene classifier is available

    # --- audio cue ---
    audio_pmax: float = 0.5       # true per-step detection probability when the agent is very close
    audio_d0: float = 4.0         # true detection falls off as exp(-d/d0)
    audio_wall_atten: float = 0.5 # extra attenuation because the agent is behind a wall
    audio_p_false: float = 0.03   # true false-alarm rate per step
    # the ROBOT's (simplified) model of its own microphone, used in the Bayes update
    model_p_hear_given_agent: float = 0.15
    model_p_hear_given_none: float = 0.03
