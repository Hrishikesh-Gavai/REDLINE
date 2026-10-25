"""The edit: time dilation, stutters, freezes — all triggered by the music.

We never change the total frame count. Each edit *overwrites* a window of frames,
then the picture jumps back to where it should be. That keeps everything locked to
the original audio, so hits still land on hits.

TIME DILATION (the centrepiece): on the biggest hits, the picture drops to ~0.3x
speed for a second or so while the audio keeps going at full speed, then SNAPS back
to real time. That mismatch — world slowed, you're not — is the "overdrive" feeling.

Returns an EditPlan:
    timeline[out_frame] -> source frame index
    tags[out_frame]     -> "DILATE" | "STUTTER" | "FREEZE" | "REVERSE"
    dilate[out_frame]   -> 0..1 intensity curve (drives trails, grade, HUD)
    snaps               -> set of frames where dilation ends (big flash + shake)
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EditPlan:
    timeline: list
    tags: dict = field(default_factory=dict)
    dilate: np.ndarray = None
    snaps: set = field(default_factory=set)

    @property
    def max_lag(self):
        return max(0, max(o - s for o, s in enumerate(self.timeline)))

    @property
    def max_lead(self):
        return max(0, max(s - o for o, s in enumerate(self.timeline)))


def build_plan(n, fps, hit_frames, energy, cfg, rng):
    plan = EditPlan(list(range(n)), {}, np.zeros(n, np.float32), set())
    if not cfg.get("enabled", True):
        return plan

    sec = lambda s: max(1, int(round(s * fps)))
    dcfg = cfg["dilation"]
    hits = sorted(hit_frames)

    # "strong" hits = loudest X% — only those can trigger dilation
    hit_energy = np.array([energy[h] for h in hits]) if hits else np.array([])
    strong_cut = np.quantile(hit_energy, 1 - dcfg["top_hits"]) if len(hits) else 1.0

    busy_until = -1
    last_dilate = -10 ** 9

    for hit in hits:
        if hit <= busy_until or hit >= n - 2:
            continue

        # ── time dilation ───────────────────────────────
        if (dcfg["enabled"] and energy[hit] >= strong_cut
                and hit - last_dilate >= sec(dcfg["cooldown"])):
            L = sec(rng.uniform(*dcfg["length"]))
            L = min(L, n - hit - 1)
            speed = dcfg["speed"]
            ramp_in, ramp_out = max(1, int(L * 0.12)), max(1, int(L * 0.2))
            for k in range(L):
                plan.timeline[hit + k] = min(n - 1, hit + int(k * speed))
                plan.tags[hit + k] = "DILATE"
                a = min(1.0, k / ramp_in, (L - k) / ramp_out)
                plan.dilate[hit + k] = a
            plan.snaps.add(hit + L)
            busy_until = hit + L
            last_dilate = hit
            continue

        # ── small edits ─────────────────────────────────
        roll = rng.random()
        c1 = cfg["stutter_chance"]
        c2 = c1 + cfg["freeze_chance"]
        c3 = c2 + cfg["reverse_chance"]
        if roll < c1:
            chunk = int(rng.integers(cfg["stutter_len"][0], cfg["stutter_len"][1] + 1))
            reps = int(rng.integers(cfg["stutter_repeats"][0], cfg["stutter_repeats"][1] + 1))
            pattern, name = [hit + (k % chunk) for k in range(chunk * reps)], "STUTTER"
        elif roll < c2:
            pattern, name = [hit] * sec(rng.uniform(*cfg["freeze_len"])), "FREEZE"
        elif roll < c3:
            L = sec(rng.uniform(*cfg["reverse_len"]))
            pattern, name = [hit + L - 1 - k for k in range(L)], "REVERSE"
        else:
            continue
        for k, src in enumerate(pattern):
            if hit + k >= n:
                break
            plan.timeline[hit + k] = min(src, n - 1)
            plan.tags[hit + k] = name
        busy_until = hit + len(pattern)

    return plan
