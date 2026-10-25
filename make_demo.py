"""Builds the before/after GIFs used in the README.

    python make_demo.py                                   # input/clip.mp4 vs output/clip_go_redline.mp4
    python make_demo.py --start 12 --seconds 6            # pick the best moment
    python make_demo.py --before input/a.mp4 --after output/a_go_redline.mp4

Writes:
    assets/demo/before.gif
    assets/demo/after.gif
    assets/demo/compare.gif    (stacked: before on top, after below)
"""
import argparse
import os
import subprocess

from src.video_io import ffmpeg_bin


def gif(ffmpeg, src, dst, start, seconds, width, fps):
    vf = (f"fps={fps},scale={width}:-2:flags=lanczos,split[a][b];"
          f"[a]palettegen=max_colors=128:stats_mode=diff[p];"
          f"[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", str(start), "-t", str(seconds),
                    "-i", src, "-filter_complex", vf, "-loop", "0", dst], check=True)


def compare(ffmpeg, before, after, dst, start, seconds, width, fps):
    vf = (f"[0:v]fps={fps},scale={width}:-2:flags=lanczos[l];"
          f"[1:v]fps={fps},scale={width}:-2:flags=lanczos[r];"
          f"[l][r]vstack=inputs=2,split[a][b];"
          f"[a]palettegen=max_colors=128:stats_mode=diff[p];"
          f"[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error",
                    "-ss", str(start), "-t", str(seconds), "-i", before,
                    "-ss", str(start), "-t", str(seconds), "-i", after,
                    "-filter_complex", vf, "-loop", "0", dst], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="input/clip.mp4")
    ap.add_argument("--after", default="output/clip_go_redline.mp4")
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--fps", type=int, default=10)
    a = ap.parse_args()

    ff = ffmpeg_bin()
    if not ff:
        raise SystemExit("ffmpeg not found — pip install imageio-ffmpeg")
    os.makedirs("assets/demo", exist_ok=True)
    gif(ff, a.before, "assets/demo/before.gif", a.start, a.seconds, a.width, a.fps)
    gif(ff, a.after, "assets/demo/after.gif", a.start, a.seconds, a.width, a.fps)
    compare(ff, a.before, a.after, "assets/demo/compare.gif", a.start, a.seconds, a.width, a.fps)
    for f in ("before", "after", "compare"):
        p = f"assets/demo/{f}.gif"
        print(f"{p}  {os.path.getsize(p) / 1e6:.1f} MB")
    print("Tip: keep each GIF under ~10 MB so GitHub loads it fast (lower --width, --fps or --seconds).")


if __name__ == "__main__":
    main()
