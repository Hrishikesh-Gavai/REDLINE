"""Finds things to put boxes on. Three sources, all return the same shape:

    Target(id, x, y, w, h, kind, label)

- FeatureTracker : corners / high-contrast points followed with optical flow.
                   This is the main source of the twitchy "everything is being scanned" boxes.
- MotionTracker  : blobs where pixels changed since last frame (frame differencing).
- YoloTracker    : real people / objects with persistent IDs (optional, needs ultralytics).
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Target:
    id: int
    x: float          # center
    y: float
    w: float
    h: float
    kind: str         # "feature" | "motion" | "yolo"
    label: str = ""   # class name for yolo
    conf: float = 0.0


class FeatureTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.prev_gray = None
        self.points = np.empty((0, 1, 2), np.float32)
        self.ids = []
        self.next_id = 0
        self.frame_i = 0

    def _detect(self, gray):
        pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.cfg["max_points"], qualityLevel=self.cfg["quality"],
            minDistance=self.cfg["min_distance"], blockSize=7,
        )
        return pts if pts is not None else np.empty((0, 1, 2), np.float32)

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is not None and len(self.points):
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.points, None, winSize=(21, 21), maxLevel=3,
            )
            keep = status.ravel() == 1
            self.points = nxt[keep].reshape(-1, 1, 2)
            self.ids = [i for i, k in zip(self.ids, keep) if k]

        # top up with fresh points periodically (or when we've lost most of them)
        if self.frame_i % self.cfg["refresh_every"] == 0 or len(self.points) < self.cfg["max_points"] // 3:
            fresh = self._detect(gray)
            if len(self.points) and len(fresh):
                # drop fresh points that sit on top of ones we're already tracking
                d = np.linalg.norm(fresh.reshape(-1, 1, 2) - self.points.reshape(1, -1, 2), axis=2)
                fresh = fresh[d.min(axis=1) > self.cfg["min_distance"]]
            room = self.cfg["max_points"] - len(self.points)
            fresh = fresh[:max(0, room)]
            if len(fresh):
                self.points = np.concatenate([self.points, fresh.reshape(-1, 1, 2)]) if len(self.points) else fresh.reshape(-1, 1, 2)
                self.ids += list(range(self.next_id, self.next_id + len(fresh)))
                self.next_id += len(fresh)

        self.prev_gray = gray
        self.frame_i += 1
        return [Target(i, float(p[0][0]), float(p[0][1]), 0, 0, "feature")
                for i, p in zip(self.ids, self.points)]


class MotionTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.prev = None

    def update(self, frame):
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 9), 0)
        if self.prev is None:
            self.prev = gray
            return []
        diff = cv2.absdiff(self.prev, gray)
        self.prev = gray
        _, mask = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, None, iterations=3)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= self.cfg["min_area"]]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[: self.cfg["max_blobs"]]
        out = []
        for k, c in enumerate(contours):
            x, y, w, h = cv2.boundingRect(c)
            out.append(Target(100000 + k, x + w / 2, y + h / 2, w, h, "motion"))
        return out


class YoloTracker:
    def __init__(self, cfg):
        from ultralytics import YOLO   # imported lazily so it's truly optional
        self.cfg = cfg
        self.model = YOLO(cfg["model"])
        self.frame_i = 0
        self.last = []

    def update(self, frame):
        if self.frame_i % self.cfg["every"] == 0:
            res = self.model.track(frame, persist=True, conf=self.cfg["conf"], verbose=False)[0]
            self.last = []
            if res.boxes is not None and len(res.boxes):
                ids = res.boxes.id.int().tolist() if res.boxes.id is not None else range(len(res.boxes))
                for tid, box, cls, conf in zip(ids, res.boxes.xywh.tolist(),
                                               res.boxes.cls.int().tolist(), res.boxes.conf.tolist()):
                    x, y, w, h = box
                    self.last.append(Target(200000 + tid, x, y, w, h, "yolo",
                                            res.names[cls], float(conf)))
        self.frame_i += 1
        return self.last


class TrackerStack:
    """Runs every enabled tracker and merges their targets."""

    def __init__(self, cfg):
        self.trackers = []
        if cfg["features"]["enabled"]:
            self.trackers.append(FeatureTracker(cfg["features"]))
        if cfg["motion"]["enabled"]:
            self.trackers.append(MotionTracker(cfg["motion"]))
        if cfg["yolo"]["enabled"]:
            try:
                self.trackers.append(YoloTracker(cfg["yolo"]))
            except ImportError:
                print("[!] yolo.enabled is true but ultralytics isn't installed — skipping YOLO.")
                print("    pip install ultralytics")

    def update(self, frame):
        targets = []
        for t in self.trackers:
            targets += t.update(frame)
        return targets
