"""
REDLINE — turn a normal clip into a beat-synced, box-tracked, glitched edit.

    python main.py input/clip.mp4
    python main.py input/clip.mp4 -o output/edit.mp4 --preview
    python main.py input/clip.mp4 --set hud.color=[60,255,60] --set tracking.yolo.enabled=true
"""
import argparse
import ast
import os
import sys

import yaml

from src.pipeline import run


def apply_override(cfg, dotted):
    """--set a.b.c=value  →  cfg['a']['b']['c'] = value  (value parsed as Python literal)."""
    key, _, raw = dotted.partition("=")
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        value = {"true": True, "false": False, "null": None}.get(raw.lower(), raw)
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node[p]
    node[parts[-1]] = value


def main():
    ap = argparse.ArgumentParser(description="Beat-synced HUD/glitch video editor")
    ap.add_argument("input", help="path to input video")
    ap.add_argument("-o", "--output", help="output path (default: output/<name>_go_redline.mp4)")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="override a config value, e.g. --set hud.show_fraction=0.6")
    ap.add_argument("--preview", action="store_true", help="also save 6 still frames to output/preview/")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for s in args.set:
        apply_override(cfg, s)

    name = os.path.splitext(os.path.basename(args.input))[0]
    out = args.output or os.path.join("output", f"{name}_go_redline.mp4")
    preview = os.path.join("output", "preview") if args.preview else None

    if not os.path.exists(args.input):
        sys.exit(f"input not found: {args.input}")
    try:
        run(args.input, out, cfg, preview_dir=preview)
    except RuntimeError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
