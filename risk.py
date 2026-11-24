"""Risk layer: (1) scene description -> risk label (LLM or fallback), (2) Bayesian belief per zone,
(3) simulated audio cue."""
import json
import os
import re
import numpy as np
from world import ZONES, ZONE_NAMES

SYSTEM_PROMPT = (
    "You assess hazards for a mobile robot driving through an indoor corridor. You are given a short "
    "description of a region the robot cannot currently see. Estimate how likely it is that a person "
    "or moving obstacle emerges from that hidden region into the robot's path within the next two "
    "seconds. Answer with JSON only: {\"risk\": \"low\" | \"medium\" | \"high\", \"reason\": \"<one sentence>\"}. "
    "low = an agent essentially cannot come from there (sealed or dead-end space). "
    "medium = possible but uncommon. high = frequent traffic or a blind crossing."
)


def _fallback(description):
    """Keyword stand-in for the LLM so experiments run offline and reproducibly."""
    d = description.lower()
    if any(k in d for k in ("no door", "no other exit", "dead end", "sealed", "nothing can enter")):
        return "low", "fallback: sealed / dead-end space"
    if any(k in d for k in ("junction", "crossing", "intersection", "busy", "blind")):
        return "high", "fallback: blind crossing with traffic"
    return "medium", "fallback: unknown"


def classify(description, use_llm=False, cache_path="results/risk_cache.json"):
    cache = {}
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path))
    if description in cache:
        return cache[description]["risk"], cache[description]["reason"], "cache"
    source = "fallback"
    if use_llm and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=os.getenv("RISK_MODEL", "claude-sonnet-5-5"), max_tokens=200, system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": description}])
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
            risk, reason, source = data["risk"].lower(), data.get("reason", ""), "llm"
            assert risk in ("low", "medium", "high")
        except Exception as e:  # network, parsing, bad label ... never crash the experiment
            print(f"[risk] LLM call failed ({e}); using fallback")
            risk, reason = _fallback(description)
    else:
        risk, reason = _fallback(description)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache[description] = {"risk": risk, "reason": reason, "source": source}
    json.dump(cache, open(cache_path, "w"), indent=2)
    return risk, reason, source


def zone_priors(cfg, mode, use_llm=False):
    """mode 'llm' -> classifier per zone; 'uniform' -> same medium prior everywhere."""
    pm = dict(cfg.prior_map)
    out = {}
    for name in ZONE_NAMES:
        if mode == "uniform":
            out[name] = cfg.uniform_prior
        else:
            risk, reason, src = classify(ZONES[name][1], use_llm)
            out[name] = pm[risk]
            print(f"[risk] {name:9s} -> {risk:6s} (prior {pm[risk]:.2f}, via {src}): {reason}")
    return np.array([out[n] for n in ZONE_NAMES])


class Belief:
    """P(an unseen agent is about to come out of zone z), one number per zone."""

    def __init__(self, cfg, priors, audio):
        self.cfg, self.prior, self.audio = cfg, priors.copy(), audio
        self.p = priors.copy()
        self.was_clear = np.zeros(len(priors), bool)

    def update(self, zone_occluded, heard):
        c = self.cfg
        for z in range(len(self.p)):
            if not zone_occluded[z]:             # we have SEEN the whole zone (or passed it): peek -> clear
                self.p[z] = c.p_clear
                self.was_clear[z] = True
                continue
            if self.was_clear[z]:                # zone became hidden again -> back to the prior
                self.p[z] = self.prior[z]
                self.was_clear[z] = False
                continue
            self.p[z] = (1 - c.leak) * self.p[z] + c.leak * self.prior[z]
            if self.audio and heard is not None:
                la = c.model_p_hear_given_agent if heard else 1 - c.model_p_hear_given_agent
                ln = c.model_p_hear_given_none if heard else 1 - c.model_p_hear_given_none
                self.p[z] = la * self.p[z] / (la * self.p[z] + ln * (1 - self.p[z]))
            self.p[z] = float(np.clip(self.p[z], max(c.p_floor, c.silence_floor * self.prior[z]), 0.95))


def simulate_audio(cfg, rng, agents, robot_pos, t, seen_flags):
    """Did the robot 'hear' something this step? Works through walls, noisy, with false alarms.
    The loudest hidden, moving agent sets the detection probability."""
    p = cfg.audio_p_false
    for a, seen in zip(agents, seen_flags):
        if a.moving(t) and not seen:
            d = np.linalg.norm(a.pos(t) - robot_pos)
            p = max(p, cfg.audio_pmax * np.exp(-d / cfg.audio_d0) * cfg.audio_wall_atten)
    return bool(rng.random() < p)
