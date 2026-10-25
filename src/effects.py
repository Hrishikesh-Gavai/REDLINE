"""Effect layers. Each one is  fn(frame, ctx, **params) -> frame.

ctx carries timing info so effects can react to the music:
    ctx.i          output frame index
    ctx.is_hit     True on a beat
    ctx.since_hit  frames since the last beat (for decays)
    ctx.energy     0..1 audio loudness at this frame
    ctx.dilate     0..1 time-dilation intensity
    ctx.is_snap    True on the frame where dilation snaps back to real time
    ctx.color      current neon color (BGR) — rotates through the palette on each hit
    ctx.fps        frames per second (so durations can be given in seconds)
    ctx.rng        seeded random generator

To add your own effect: write a function below, add it to REGISTRY,
then add an entry with the same `name` to the effects list in config.yaml.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Ctx:
    i: int
    is_hit: bool
    since_hit: int
    energy: float
    dilate: float
    is_snap: bool
    since_snap: int
    color: tuple
    color2: tuple
    hit_count: int
    fps: float
    rng: np.random.Generator


def _frames(sec, fps):
    return max(1, int(round(sec * fps)))


# ── neon flashes ─────────────────────────────────────────
def neon_flash(f, ctx, strength=0.55, decay=0.08, snap_strength=0.9, **_):
    """Screen-blends the current neon color over the frame on hits (instead of plain white).
    Screen keeps darks dark-ish and makes highlights glow — reads as light, not paint."""
    d = _frames(decay, ctx.fps)
    if ctx.since_snap <= d:
        k = snap_strength * (1 - ctx.since_snap / (d + 1))
    elif ctx.since_hit <= d:
        k = strength * (1 - ctx.since_hit / (d + 1)) * (0.6 + 0.4 * ctx.energy)
    else:
        return f
    layer = np.empty_like(f)
    layer[:] = np.array(ctx.color, np.float32) * k
    return cv2.subtract(cv2.add(f, layer), cv2.multiply(f, layer, scale=1 / 255))


def duotone_slam(f, ctx, chance=0.18, length=0.05, **_):
    """On some hits: 1–3 frames of hard two-color graphic look (dark color → neon color).
    Very 'motion design' — the frame becomes a flat poster for an instant."""
    L = _frames(length, ctx.fps)
    if not (ctx.since_hit < L or ctx.since_snap < L):
        return f
    # decide once per hit (seeded by hit index so it's stable across its frames)
    if ctx.since_snap >= L and np.random.default_rng(ctx.hit_count * 7919).random() > chance:
        return f
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    g = np.clip((g - 0.15) * 1.6, 0, 1)[..., None]
    dark = np.array(ctx.color2, np.float32) * 0.18
    light = np.array(ctx.color, np.float32)
    return (dark * (1 - g) + light * g).astype(np.uint8)


def graphic_slam(f, ctx, chance=0.12, length=0.12, words=("SYNC",), **_):
    """On some hits: a bold word slams in on a color bar with small technical text.
    Inspired by graphic-design title cards — big type, flat color, hard cuts."""
    L = _frames(length, ctx.fps)
    if ctx.since_hit >= L:
        return f
    r = np.random.default_rng(ctx.hit_count * 104729)
    if r.random() > chance:
        return f
    h, w = f.shape[:2]
    word = words[int(r.integers(len(words)))]
    out = f.copy()
    scale = w / 520
    th = max(2, int(scale * 3))
    (tw, tht), _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_DUPLEX, scale, th)
    y = int(h * r.uniform(0.3, 0.7))
    x = int(r.uniform(0.05, max(0.06, 0.95 - tw / w)) * w)
    # color bar slides in over the first few frames
    grow = min(1.0, (ctx.since_hit + 1) / max(1, L // 3))
    bar_w = int((tw + 40) * grow)
    cv2.rectangle(out, (x - 20, y - tht - 18), (x - 20 + bar_w, y + 18), ctx.color, -1)
    if grow >= 1:
        cv2.putText(out, word, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, (10, 10, 10), th, cv2.LINE_AA)
        small = f"{r.integers(1000, 9999)}-{r.integers(10, 99)} // {r.uniform(0, 1):.4f}"
        cv2.putText(out, small, (x, y + 18 + int(16 * scale)), cv2.FONT_HERSHEY_PLAIN,
                    max(1, scale * 0.9), ctx.color, 1, cv2.LINE_AA)
    return out


# ── motion ───────────────────────────────────────────────
def snap_shake(f, ctx, pixels=18, length=0.1, zoom=0.04, **_):
    """When dilation snaps back: a quick punch-in zoom + shake, like the world catching up."""
    L = _frames(length, ctx.fps)
    if ctx.since_snap >= L:
        return f
    k = 1 - ctx.since_snap / L
    h, w = f.shape[:2]
    dx = int(ctx.rng.integers(-pixels, pixels + 1) * k)
    dy = int(ctx.rng.integers(-pixels, pixels + 1) * k)
    z = 1 + zoom * k
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, z)
    M[:, 2] += (dx, dy)
    return cv2.warpAffine(f, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def dilation_breathe(f, ctx, zoom=0.03, **_):
    """Slow push-in while time is dilated — makes the slow-mo feel heavy."""
    if ctx.dilate <= 0.02:
        return f
    h, w = f.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1 + zoom * ctx.dilate)
    return cv2.warpAffine(f, M, (w, h), borderMode=cv2.BORDER_REFLECT)


# ── glitch ───────────────────────────────────────────────
def rgb_split(f, ctx, offset=2, beat_offset=8, dilate_offset=6, **_):
    o = offset
    if ctx.since_hit <= 1 or ctx.since_snap <= 1:
        o = beat_offset
    o = int(max(o, dilate_offset * ctx.dilate))
    if o <= 0:
        return f
    b, g, r = cv2.split(f)
    return cv2.merge([np.roll(b, -o, axis=1), g, np.roll(r, o, axis=1)])


def glitch_slices(f, ctx, chance=0.02, on_hits=0.3, slices=(2, 6), max_shift=40, **_):
    hit_fire = ctx.is_hit and np.random.default_rng(ctx.hit_count * 31).random() < on_hits
    if not (hit_fire or ctx.is_snap or ctx.rng.random() < chance):
        return f
    out = f.copy()
    h = f.shape[0]
    for _ in range(int(ctx.rng.integers(slices[0], slices[1] + 1))):
        y = int(ctx.rng.integers(0, h - 4))
        sh = int(ctx.rng.integers(2, max(3, h // 14)))
        out[y:y + sh] = np.roll(f[y:y + sh], int(ctx.rng.integers(-max_shift, max_shift + 1)), axis=1)
        # tint the displaced slice with the neon color
        out[y:y + sh] = cv2.addWeighted(out[y:y + sh], 0.75,
                                        np.full_like(out[y:y + sh], ctx.color), 0.25, 0)
    return out


def pixelate(f, ctx, chance=0.01, block=(6, 14), **_):
    if ctx.rng.random() >= chance:
        return f
    h, w = f.shape[:2]
    b = int(ctx.rng.integers(block[0], block[1] + 1))
    small = cv2.resize(f, (max(1, w // b), max(1, h // b)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def posterize(f, ctx, levels=4, **_):
    step = 256 // max(2, levels)
    return (f // step * step + step // 2).astype(np.uint8)


# ── finish ───────────────────────────────────────────────
def scanlines(f, ctx, spacing=3, darkness=0.12, **_):
    out = f.copy()
    out[::spacing] = (out[::spacing] * (1 - darkness)).astype(np.uint8)
    return out


_grain_bank = {}


def grain(f, ctx, amount=6, **_):
    """Pre-generates a few noise tiles once and cycles them — far cheaper than fresh noise per frame."""
    h, w = f.shape[:2]
    key = (h, w, amount)
    if key not in _grain_bank:
        r = np.random.default_rng(0)
        tiles = []
        for _ in range(8):
            n = r.normal(0, amount, (h // 2, w // 2)).astype(np.float32)
            n = cv2.resize(n, (w, h), interpolation=cv2.INTER_NEAREST)
            tiles.append((np.clip(n, 0, 255).astype(np.uint8), np.clip(-n, 0, 255).astype(np.uint8)))
        _grain_bank[key] = tiles
    pos, neg = _grain_bank[key][int(ctx.rng.integers(8))]
    pos3, neg3 = cv2.merge([pos] * 3), cv2.merge([neg] * 3)
    return cv2.subtract(cv2.add(f, pos3), neg3)


_vignette_cache = {}


def vignette(f, ctx, strength=0.4, dilate_extra=0.35, **_):
    h, w = f.shape[:2]
    s = round(strength + dilate_extra * ctx.dilate, 2)    # quantized so masks can be cached
    key = (h, w, s)
    if key not in _vignette_cache:
        y, x = np.ogrid[:h, :w]
        d = ((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2
        m = np.clip(1 - s * d, 0, 1).astype(np.float32)
        _vignette_cache[key] = cv2.merge([m, m, m])
        if len(_vignette_cache) > 64:
            _vignette_cache.pop(next(iter(_vignette_cache)))
    return cv2.multiply(f, _vignette_cache[key], dtype=cv2.CV_8U)


REGISTRY = {
    "neon_flash": neon_flash,
    "duotone_slam": duotone_slam,
    "graphic_slam": graphic_slam,
    "snap_shake": snap_shake,
    "dilation_breathe": dilation_breathe,
    "rgb_split": rgb_split,
    "glitch_slices": glitch_slices,
    "pixelate": pixelate,
    "posterize": posterize,
    "scanlines": scanlines,
    "grain": grain,
    "vignette": vignette,
}


def build_stack(effect_cfgs):
    stack = []
    for e in effect_cfgs:
        if not e.get("enabled", True):
            continue
        name = e["name"]
        if name not in REGISTRY:
            print(f"[!] unknown effect '{name}' — skipping")
            continue
        params = {k: v for k, v in e.items() if k not in ("name", "enabled")}
        stack.append((REGISTRY[name], params))
    return stack


def apply_stack(frame, stack, ctx):
    for fn, params in stack:
        frame = fn(frame, ctx, **params)
    return frame
