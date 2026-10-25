"""Generates a test clip (moving shapes + a kick drum at 120 bpm)
so you can try the pipeline before using real footage.

    python make_test_clip.py                      → 6s 960x540 30fps
    python make_test_clip.py 1080 1080 60 60      → 60s 1080x1080 60fps (stress test)
"""
import os
import subprocess
import wave

import cv2
import numpy as np

import sys
a = [int(x) for x in sys.argv[1:5]]
W, H, FPS, SECS = (a + [960, 540, 30, 6][len(a):])
BPM = 120
os.makedirs("input", exist_ok=True)

# video: drifting circles/rects on a textured background
rng = np.random.default_rng(1)
bg = cv2.GaussianBlur((rng.random((H, W, 3)) * 80).astype(np.uint8), (0, 0), 6)
shapes = [dict(x=rng.uniform(100, W - 100), y=rng.uniform(100, H - 100),
               vx=rng.uniform(-8, 8) * 30 / FPS, vy=rng.uniform(-6, 6) * 30 / FPS, r=int(rng.integers(25, 70)),
               c=tuple(int(v) for v in rng.integers(80, 255, 3)), sq=bool(rng.random() < .5))
          for _ in range(7)]
vw = cv2.VideoWriter("input/_v.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
for i in range(FPS * SECS):
    f = bg.copy()
    cv2.putText(f, "TEST", (W // 2 - 120, H // 2 + 30), cv2.FONT_HERSHEY_DUPLEX, 3, (200, 200, 200), 6)
    for s in shapes:
        s["x"] += s["vx"]; s["y"] += s["vy"]
        if not s["r"] < s["x"] < W - s["r"]: s["vx"] *= -1
        if not s["r"] < s["y"] < H - s["r"]: s["vy"] *= -1
        p = (int(s["x"]), int(s["y"]))
        if s["sq"]:
            cv2.rectangle(f, (p[0] - s["r"], p[1] - s["r"]), (p[0] + s["r"], p[1] + s["r"]), s["c"], -1)
        else:
            cv2.circle(f, p, s["r"], s["c"], -1)
    vw.write(f)
vw.release()

# audio: kick on every beat + hat in between
sr = 22050
t = np.arange(sr * SECS) / sr
y = np.zeros_like(t)
beat = 60 / BPM
for k in range(int(SECS / beat)):
    s = int(k * beat * sr); n = int(0.25 * sr)
    tt = np.arange(n) / sr
    y[s:s + n] += np.sin(2 * np.pi * (50 + 120 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 12)
    h = int((k + .5) * beat * sr); m = int(0.04 * sr)
    y[h:h + m] += rng.normal(0, .25, len(y[h:h + m])) * np.exp(-np.arange(len(y[h:h + m])) / sr * 80)
y = (np.clip(y, -1, 1) * 30000).astype(np.int16)
with wave.open("input/_a.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(y.tobytes())

subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "input/_v.mp4", "-i", "input/_a.wav",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                "input/test_clip.mp4"], check=True)
os.remove("input/_v.mp4"); os.remove("input/_a.wav")
print("wrote input/test_clip.mp4")
