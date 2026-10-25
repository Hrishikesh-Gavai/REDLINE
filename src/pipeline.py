"""Glues everything together:

  probe → beats → edit plan → for each frame (streamed):
      fetch source frame → overdrive (trails / glow / grade) → track → HUD → effects → write
  → put the original audio back
"""
import os
import tempfile
import time

import numpy as np

from . import audio
from .editor import build_plan
from .effects import Ctx, apply_stack, build_stack
from .hud import HUD
from .overdrive import Overdrive
from .palette import Palette
from .tracking import TrackerStack
from .video_io import FrameSource, VideoWriter, extract_audio, ffmpeg_bin, mux_audio


def run(input_path, output_path, cfg, preview_dir=None):
    if not ffmpeg_bin():
        raise RuntimeError(
            "ffmpeg not found. Easiest fix:  pip install imageio-ffmpeg\n"
            "  (or install system ffmpeg: winget install Gyan.FFmpeg, then reopen the terminal)")
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    t0 = time.time()
    rng = np.random.default_rng(cfg.get("seed"))
    vcfg = cfg["video"]

    # 1. open (streamed — nothing loaded yet)
    print(f"[1/5] opening {input_path}")
    src = FrameSource(input_path, vcfg.get("max_width"), vcfg.get("start", 0.0), vcfg.get("duration"))
    fps, n = src.fps, src.n
    print(f"      {n} frames · {src.width}x{src.height} · {fps:.2f} fps · {src.duration:.1f}s")

    # 2. beats
    print("[2/5] finding beats")
    acfg = cfg["audio"]
    tmp = tempfile.mkdtemp()
    wav = None
    if acfg.get("enabled", True):
        a_src = acfg.get("external") or input_path
        wav = extract_audio(a_src, os.path.join(tmp, "a.wav"),
                            0.0 if acfg.get("external") else src.start, src.duration)
    if wav:
        hits, energy = audio.detect_hit_frames(wav, fps, n, acfg.get("onset_strength", 1.3))
        print(f"      {len(hits)} hits detected")
    else:
        hits, energy = audio.fake_hits(n, fps)
        print(f"      no audio found — using a 120 bpm grid ({len(hits)} hits)")
    # optional thinning: keep hits at least min_gap apart (less chaotic on busy tracks)
    gap = int(round(acfg.get("min_hit_gap", 0.0) * fps))
    if gap > 1:
        kept, last = set(), -10 ** 9
        for h in sorted(hits):
            if h - last >= gap:
                kept.add(h)
                last = h
        hits = kept
        print(f"      {len(hits)} hits after spacing")

    # 3. plan the edit
    print("[3/5] planning the edit")
    plan = build_plan(n, fps, hits, energy, cfg["edit"], rng)
    n_dil = sum(1 for v in plan.tags.values() if v == "DILATE")
    print(f"      {len(plan.snaps)} time-dilations ({n_dil / fps:.1f}s total) · "
          f"{len(plan.tags) - n_dil} frames of stutter/freeze/reverse")
    keep_behind = plan.max_lag + 2

    # 4. render
    print("[4/5] rendering")
    palette = Palette(cfg["palette"])
    trackers = TrackerStack(cfg["tracking"])
    hud = HUD(cfg["hud"], rng, src.width, src.height, fps)
    od = Overdrive(cfg["overdrive"], palette, src.width, src.height)
    stack = build_stack(cfg["effects"])
    silent_path = os.path.join(tmp, "silent.mp4")
    writer = VideoWriter(silent_path, src.width, src.height, fps, cfg["output"]["crf"])
    ocfg = cfg["overdrive"]

    preview_at = set()
    if preview_dir:
        # grab stills mid-dilation + on hits + spread out, so you see every look
        preview_at |= {s - int(0.4 * fps) for s in sorted(plan.snaps)[:3]}
        preview_at |= set(sorted(hits)[1:4])
        preview_at |= set(np.linspace(0, n - 1, 4, dtype=int).tolist())

    since_hit = since_snap = 10 ** 6
    hit_count = 0
    t_render = time.time()
    for out_i in range(n):
        s_i = plan.timeline[out_i]
        frame = src.get(s_i).copy()
        src.evict_before(out_i - keep_behind)

        is_hit = out_i in hits
        is_snap = out_i in plan.snaps
        if is_hit or is_snap:
            hit_count += 1
        since_hit = 0 if is_hit else since_hit + 1
        since_snap = 0 if is_snap else since_snap + 1
        e = float(energy[out_i])
        d = float(plan.dilate[out_i])
        color = palette.at(hit_count)
        color2 = palette.at(hit_count + 1)

        # tracking runs on the CLEAN frame so boxes follow real features
        targets = trackers.update(frame)

        od.push(s_i, src.get(s_i))
        pulse = ocfg["trails"]["on_hits"] * max(0.0, 1 - since_hit / max(1, 0.15 * fps))
        frame = od.dilation_grade(frame, d)
        frame = od.trails(frame, max(d, pulse), hit_count)
        frame = od.edge_glow(frame, max(d * ocfg["edge_glow"]["in_dilation"],
                                        pulse * ocfg["edge_glow"]["on_hits"]), color)

        frame = hud.draw(frame, targets, out_i, is_hit, e, plan.tags.get(out_i), d, accent=color)
        frame = apply_stack(frame, stack, Ctx(out_i, is_hit, since_hit, e, d, is_snap, since_snap,
                                              color, color2, hit_count, fps, rng))
        writer.write(frame)

        if out_i in preview_at:
            import cv2
            os.makedirs(preview_dir, exist_ok=True)
            cv2.imwrite(os.path.join(preview_dir, f"frame_{out_i:05d}.jpg"), frame)
        if out_i % 30 == 0 and out_i:
            el = time.time() - t_render
            eta = el / out_i * (n - out_i)
            print(f"      {out_i:5d}/{n}  ({100 * out_i / n:4.1f}%)  eta {eta / 60:4.1f} min   ", end="\r")
    writer.close()
    src.close()
    print(f"      {n}/{n}  (100.0%)                       ")

    # 5. audio back on
    print("[5/5] muxing audio")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    audio_src = acfg.get("external") or input_path
    ok = cfg["output"].get("keep_audio", True) and mux_audio(
        silent_path, audio_src, output_path,
        0.0 if acfg.get("external") else src.start, src.duration)
    if not ok:
        os.replace(silent_path, output_path)
        print("      (no audio track — saved video only)")

    print(f"done → {output_path}  [{(time.time() - t0) / 60:.1f} min]")
