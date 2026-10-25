"""The overlay: flickering boxes, random numbers, connecting lines, corner readouts.

The flicker is what sells it. Every target gets a tiny random 'life' — it blinks on
for a few frames with a random style + size, then blinks off. On beats, way more
targets switch on at once, so the whole frame lights up in sync with the music.
"""
import string

import cv2
import numpy as np

from .palette import hex_to_bgr

FONT = cv2.FONT_HERSHEY_PLAIN


class HUD:
    def __init__(self, cfg, rng, width, height, fps):
        self.cfg = cfg
        self.rng = rng
        self.w, self.h, self.fps = width, height, fps
        self.color = hex_to_bgr(cfg["color"])
        self.accent = hex_to_bgr(cfg["accent"])
        self.res = height / 540.0                 # scale text/lines with resolution
        self.tscale = fps / 30.0                  # lifespans are tuned for 30fps
        self.cfg = dict(cfg, font_scale=cfg["font_scale"] * max(1.0, self.res * 0.85),
                        thickness=max(cfg["thickness"], int(round(self.res * 0.9))))
        cfg = self.cfg
        self.alive = {}   # target id -> {life, style, size, label_type}
        self.pil_font = None
        if cfg.get("font_path"):
            try:
                from PIL import ImageFont
                self.pil_font = ImageFont.truetype(cfg["font_path"], max(10, int(30 * cfg["font_scale"])))
            except Exception as e:
                print(f"[!] couldn't load font {cfg['font_path']}: {e} — using OpenCV font")

    # ── state ────────────────────────────────────────────────
    def _tick(self, targets, is_hit, energy, dilate):
        cfg, rng = self.cfg, self.rng
        frac = cfg["beat_boost"] if is_hit else cfg["show_fraction"] * (0.6 + 0.8 * energy)
        frac *= (1 - 0.6 * dilate)                # calmer during slow-mo: fewer new boxes...
        current = {t.id for t in targets}
        age = 1 - 0.75 * dilate                   # ...and existing ones linger (time is slow)

        # age out / drop boxes whose target disappeared
        for tid in list(self.alive):
            self.alive[tid]["life"] -= age
            if self.alive[tid]["life"] <= 0 or tid not in current:
                del self.alive[tid]

        # spawn new ones
        for t in targets:
            if t.id in self.alive:
                continue
            p = 1.0 if t.kind == "yolo" else frac * 0.5   # yolo boxes stay steadier
            if rng.random() < p:
                lo, hi = cfg["lifespan"]
                self.alive[t.id] = {
                    "life": int(rng.integers(lo, hi + 1)) * self.tscale * (3 if t.kind == "yolo" else 1),
                    "style": rng.choice(cfg["box_styles"]),
                    "size": int(rng.integers(*cfg["feature_box_size"]) * max(1.0, self.res * 0.8)),
                    "label": rng.choice(cfg["label_types"]) if rng.random() < cfg["label_chance"] else None,
                }

    # ── text helpers ─────────────────────────────────────────
    def _random_label(self, kind, t):
        rng = self.rng
        if kind == "id":
            return f"ID{t.id % 10000:04d}"
        if kind == "coords":
            return f"{int(t.x):04d},{int(t.y):04d}"
        if kind == "hex":
            return "0x" + "".join(rng.choice(list("0123456789ABCDEF"), 6))
        if kind == "conf":
            return f"{rng.uniform(0.41, 0.99):.2f}"
        if kind == "float":
            return f"{rng.uniform(-999, 999):+.3f}"
        return "".join(rng.choice(list(string.ascii_uppercase), 4))

    def _text(self, img, txt, x, y, color, scale=None):
        scale = scale or self.cfg["font_scale"]
        x, y = int(np.clip(x, 0, self.w - 10)), int(np.clip(y, 10, self.h - 2))
        if self.pil_font is not None:
            self._pending_text.append((txt, x, y, color))
        else:
            cv2.putText(img, txt, (x, y), FONT, scale * 2.4, color, 1, cv2.LINE_AA)

    def _flush_pil_text(self, img):
        if not self._pending_text:
            return img
        from PIL import Image, ImageDraw
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for txt, x, y, c in self._pending_text:
            d.text((x, y - 10), txt, font=self.pil_font, fill=(c[2], c[1], c[0]))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # ── shapes ───────────────────────────────────────────────
    def _box(self, img, x0, y0, x1, y1, style, color):
        th = self.cfg["thickness"]
        x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
        if style == "rect":
            cv2.rectangle(img, (x0, y0), (x1, y1), color, th, cv2.LINE_AA)
        elif style == "brackets":
            L = max(3, (x1 - x0) // 4)
            for (cx, cy, dx, dy) in [(x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)]:
                cv2.line(img, (cx, cy), (cx + dx * L, cy), color, th, cv2.LINE_AA)
                cv2.line(img, (cx, cy), (cx, cy + dy * L), color, th, cv2.LINE_AA)
        elif style == "cross":
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            cv2.rectangle(img, (x0, y0), (x1, y1), color, th, cv2.LINE_AA)
            cv2.line(img, (cx, y0 - 4), (cx, y1 + 4), color, th, cv2.LINE_AA)
            cv2.line(img, (x0 - 4, cy), (x1 + 4, cy), color, th, cv2.LINE_AA)
        elif style == "filled":
            # inverted patch — the box 'cuts out' the image under it
            x0c, y0c = max(0, x0), max(0, y0)
            x1c, y1c = min(self.w, x1), min(self.h, y1)
            if x1c > x0c and y1c > y0c:
                img[y0c:y1c, x0c:x1c] = 255 - img[y0c:y1c, x0c:x1c]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, th, cv2.LINE_AA)

    # ── main ─────────────────────────────────────────────────
    def draw(self, frame, targets, out_i, is_hit, energy, edit_tag=None, dilate=0.0, accent=None):
        if accent is not None:
            self.accent = accent
        self._tick(targets, is_hit, energy, dilate)
        self._dilate = dilate
        self._pending_text = []
        img = frame
        by_id = {t.id: t for t in targets}
        visible = []

        for tid, st in self.alive.items():
            t = by_id.get(tid)
            if t is None:
                continue
            if t.kind == "feature":
                s = st["size"]
                x0, y0, x1, y1 = t.x - s / 2, t.y - s / 2, t.x + s / 2, t.y + s / 2
                color = self.color
            else:
                x0, y0, x1, y1 = t.x - t.w / 2, t.y - t.h / 2, t.x + t.w / 2, t.y + t.h / 2
                color = self.accent if t.kind == "yolo" else self.color
            style = "rect" if t.kind == "yolo" else st["style"]
            self._box(img, x0, y0, x1, y1, style, color)
            visible.append((t.x, t.y))

            if t.kind == "yolo":
                self._text(img, f"{t.label.upper()} {t.conf:.2f}", x0, y0 - 4, color)
                self._text(img, f"ID{t.id - 200000:03d}", x0, y1 + 12, color)
            elif st["label"]:
                self._text(img, self._random_label(st["label"], t), x1 + 3, y0 + 8, color)

        if self.cfg["connect_lines"] and len(visible) > 1:
            self._connect(img, visible)
        if self.cfg["corner_readout"]:
            self._readout(img, out_i, len(visible), energy, is_hit, edit_tag)

        return self._flush_pil_text(img) if self.pil_font else img

    def _connect(self, img, pts):
        pts = np.array(pts)
        d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
        np.fill_diagonal(d, np.inf)
        pairs = np.argwhere(d < self.cfg["connect_dist"])
        pairs = pairs[pairs[:, 0] < pairs[:, 1]]
        if len(pairs) > self.cfg["max_lines"]:
            pairs = pairs[self.rng.choice(len(pairs), self.cfg["max_lines"], replace=False)]
        for a, b in pairs:
            cv2.line(img, tuple(map(int, pts[a])), tuple(map(int, pts[b])), self.accent, 1, cv2.LINE_AA)

    def _readout(self, img, i, n_visible, energy, is_hit, edit_tag):
        c = self.color
        r = max(1.0, self.res * 0.85)
        lh, m = int(16 * r), int(14 * r)          # line height, margin
        cw = int(8.5 * r)                         # approx char width for right-aligning
        right = lambda txt: self.w - m - len(txt) * cw

        t = i / self.fps
        tc = f"{int(t // 60):02d}:{int(t % 60):02d}:{int((t % 1) * self.fps):02d}"
        self._text(img, f"FRM {i:05d}", m, m + lh, c)
        self._text(img, f"TC  {tc}", m, m + 2 * lh, c)
        obj = f"OBJ {n_visible:03d}"
        self._text(img, obj, right(obj), m + lh, c)
        bar = int(energy * 10)
        sig = "SIG " + "|" * bar + "." * (10 - bar)
        self._text(img, sig, right(sig), m + 2 * lh, c)

        d = getattr(self, "_dilate", 0.0)
        if d > 0.02:
            # time-dilation readout: speed factor + filling bar, in the accent color
            speed = 1 - 0.7 * d
            txt = f"T-DILATION  x{speed:.2f}"
            self._text(img, txt, m, self.h - m - lh, self.accent)
            bw = int(160 * r)
            y = self.h - m - int(6 * r)
            cv2.rectangle(img, (m, y - int(4 * r)), (m + bw, y), self.accent, 1)
            cv2.rectangle(img, (m, y - int(4 * r)), (m + int(bw * d), y), self.accent, -1)
        elif is_hit:
            self._text(img, "// HIT", m, self.h - m, self.accent)
        if edit_tag and edit_tag != "DILATE":
            tag = f"[{edit_tag}]"
            self._text(img, tag, right(tag), self.h - m, self.accent)
