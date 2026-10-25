"""Reading frames in (streamed, bounded memory), writing frames out, putting audio back on."""
import os
import shutil
import subprocess

import cv2


def ffmpeg_bin():
    """System ffmpeg if it's on PATH, otherwise the copy bundled with imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class FrameSource:
    """Streams frames from disk with a small cache instead of loading the whole clip.

    The edit timeline only ever looks a little behind (slow-mo lag) or a little ahead
    (reverse chunks) of the current frame, so we keep just that window in RAM.
    A 1-minute 1080x1080 60fps clip needs ~12 GB fully loaded — this needs a few hundred MB.
    """

    def __init__(self, path, max_width=None, start=0.0, duration=None):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        scale = max_width / w if (max_width and w > max_width) else 1.0
        self.width = int(w * scale) // 2 * 2          # even dims keep H.264 happy
        self.height = int(h * scale) // 2 * 2

        self.start = start
        start_frame = int(round(start * self.fps))
        if start_frame:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        n = max(0, total - start_frame) if total > 0 else 0
        if duration:
            n = min(n, int(duration * self.fps)) if n else int(duration * self.fps)
        if n <= 0:
            raise RuntimeError(f"Couldn't read frame count from {path}")
        self.n = n
        self.duration = n / self.fps

        self.cache = {}
        self.next_read = 0
        self.last_good = None

    def _read_one(self):
        ok, frame = self.cap.read()
        if not ok:                       # frame count was slightly optimistic — repeat last
            if self.last_good is None:
                raise RuntimeError(f"No frames read from {self.path}")
            return self.last_good
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        self.last_good = frame
        return frame

    def get(self, i):
        i = max(0, min(i, self.n - 1))
        while self.next_read <= i:
            self.cache[self.next_read] = self._read_one()
            self.next_read += 1
        return self.cache[i] if i in self.cache else self.last_good

    def evict_before(self, i):
        for k in [k for k in self.cache if k < i]:
            del self.cache[k]

    def close(self):
        self.cap.release()


class VideoWriter:
    """Writes raw frames straight into ffmpeg (H.264) — better quality than cv2's writer."""

    def __init__(self, path, width, height, fps, crf=18):
        self.path = path
        cmd = [
            ffmpeg_bin(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps}",
            "-i", "-",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-pix_fmt", "yuv420p", path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame):
        self.proc.stdin.write(frame.tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def extract_audio(src, dst_wav, start=0.0, duration=None):
    """Pulls the audio track out as a mono wav (for beat detection). Returns path or None."""
    cmd = [ffmpeg_bin(), "-y", "-loglevel", "error", "-ss", str(start), "-i", src]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-vn", "-ac", "1", "-ar", "22050", dst_wav]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not os.path.exists(dst_wav) or os.path.getsize(dst_wav) < 1000:
        return None
    return dst_wav


def mux_audio(video_path, audio_src, out_path, start=0.0, duration=None):
    """Copies the processed video and lays the original audio back underneath it."""
    cmd = [ffmpeg_bin(), "-y", "-loglevel", "error", "-i", video_path, "-ss", str(start), "-i", audio_src]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", out_path]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0
