"""The 'neural overdrive' layer: afterimage trails, neon edge glow, dilation grade.

AFTERIMAGE TRAILS
    We keep a short history of source frames. For each older frame ("echo"), we find
    where it DIFFERS from now (= where something moved), tint just those pixels with a
    neon color, and screen-blend them on top. Result: colored ghost silhouettes trailing
    anything that moves — strongest during time dilation, faint pulses on normal hits.

EDGE GLOW
    Canny edges → thickened → colored → blurred → added back. Gives subjects a neon outline.

DILATION GRADE
    While time is dilated: desaturate and cool the world slightly, so the neon trails pop.

Speed: all mask/color work happens at reduced resolution; we blend into the full-res
frame only once at the end. That's what makes 1080p60 practical.
"""
from collections import deque

import cv2
import numpy as np

from .palette import hex_to_bgr


class Overdrive:
    def __init__(self, cfg, palette, width, height):
        self.cfg = cfg
        self.palette = palette
        self.w, self.h = width, height
        tcfg = cfg["trails"]
        self.hist = deque(maxlen=tcfg["echoes"] * tcfg["spacing"] + 2)
        self.last_src = None
        long_side = max(width, height)
        self.s = 540 / long_side if long_side > 540 else 1.0       # work size ≈ 540px
        self.sw, self.sh = int(width * self.s), int(height * self.s)
        self.tint = np.array(hex_to_bgr(cfg["grade"]["tint"]), np.float32) / 255.0

    def _small(self, frame):
        return cv2.resize(frame, (self.sw, self.sh), interpolation=cv2.INTER_AREA) if self.s != 1 else frame

    def push(self, src_idx, frame):
        """Store DISTINCT source frames only (slow-mo repeats them), pre-shrunk."""
        if src_idx != self.last_src:
            small = self._small(frame)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            self.hist.append((gray, gray.astype(np.float32) / 255.0))
            self.last_src = src_idx

    # ── trails ────────────────────────────────────────────
    def trails(self, f, strength, color_offset):
        tcfg = self.cfg["trails"]
        if strength <= 0.02 or len(self.hist) < tcfg["spacing"] + 1:
            return f
        cur_gray = self.hist[-1][0]
        n_echo = tcfg["echoes"]
        acc_c = np.zeros((self.sh, self.sw, 3), np.float32)   # accumulated ghost color
        acc_a = np.zeros((self.sh, self.sw), np.float32)      # accumulated coverage

        for j in range(n_echo, 0, -1):                        # far → near (newest on top)
            idx = -1 - j * tcfg["spacing"]
            if -idx > len(self.hist):
                continue
            e_gray, e_luma = self.hist[idx]
            diff = cv2.absdiff(e_gray, cur_gray)
            mask = cv2.GaussianBlur((diff > tcfg["threshold"]).astype(np.float32), (0, 0), 2)
            fade = (1 - (j - 1) / n_echo) ** 1.3
            a = mask * (fade * strength * tcfg["opacity"])
            color = np.array(self.palette.at(color_offset + j), np.float32)
            tinted = np.minimum(e_luma * 1.4, 1.0)[..., None] * color
            a3 = a[..., None]
            acc_c = acc_c * (1 - a3) + tinted * a3
            acc_a = acc_a + a * (1 - acc_a)

        if acc_a.max() < 0.01:
            return f
        # acc_c is already premultiplied by coverage → one screen blend at full res:
        #   screen(f, L) = f + L - f*L/255   (all uint8, saturating — fast)
        layer = cv2.resize(np.clip(acc_c, 0, 255).astype(np.uint8), (self.w, self.h))
        return cv2.subtract(cv2.add(f, layer), cv2.multiply(f, layer, scale=1 / 255))

    # ── edge glow ─────────────────────────────────────────
    def edge_glow(self, f, strength, color):
        gcfg = self.cfg["edge_glow"]
        if strength <= 0.02:
            return f
        gray = cv2.GaussianBlur(cv2.cvtColor(self._small(f), cv2.COLOR_BGR2GRAY), (5, 5), 0)
        edges = cv2.dilate(cv2.Canny(gray, gcfg["low"], gcfg["high"]), None, iterations=1)
        e = edges.astype(np.float32) / 255
        glow = np.minimum(cv2.GaussianBlur(e, (0, 0), gcfg["blur"] * self.s * 2) * 2.5 + e * 0.6, 1.0)
        k = strength * gcfg["intensity"]
        layer = (glow[..., None] * (np.array(color, np.float32) * k)).astype(np.uint8)
        layer = cv2.resize(layer, (self.w, self.h))
        return cv2.add(f, layer)                       # saturating add, uint8 → fast

    # ── grade ─────────────────────────────────────────────
    def dilation_grade(self, f, strength):
        gcfg = self.cfg["grade"]
        if strength <= 0.02:
            return f
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        cool = cv2.merge([cv2.convertScaleAbs(gray, alpha=1.15 * t) for t in self.tint])
        k = float(strength * gcfg["amount"])
        return cv2.addWeighted(f, 1 - k, cool, k, 0)
