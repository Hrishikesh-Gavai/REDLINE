"""Beat / onset detection → a set of frame indices where 'hits' happen.

Everything rhythmic in the edit (stutters, flashes, box bursts) keys off these frames.
Uses librosa when installed; otherwise falls back to a simple numpy energy detector.
"""
import numpy as np


def detect_hit_frames(wav_path, fps, n_frames, strength=1.3):
    """Returns (hit_frames: set[int], energy: np.ndarray[n_frames] in 0..1)."""
    try:
        import librosa
        y, sr = librosa.load(wav_path, sr=22050, mono=True)
        return _librosa_hits(y, sr, fps, n_frames, strength)
    except ImportError:
        y, sr = _load_wav_numpy(wav_path)
        return _numpy_hits(y, sr, fps, n_frames, strength)


def _librosa_hits(y, sr, fps, n_frames, strength):
    import librosa
    hop = 512
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    # onset peaks = drum hits, snaps, sharp transients
    onsets = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop,
                                        delta=0.07 * strength, units="time")
    # beat grid = steady pulse; merge both so the edit feels on-tempo AND reactive
    _, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=hop, units="time")

    times = np.concatenate([onsets, beats])
    hits = {int(round(t * fps)) for t in times if 0 <= t * fps < n_frames}

    # per-frame energy curve (used to scale effect intensity smoothly)
    env_t = librosa.times_like(env, sr=sr, hop_length=hop)
    frame_t = np.arange(n_frames) / fps
    energy = np.interp(frame_t, env_t, env)
    energy = energy / (energy.max() + 1e-8)
    return hits, energy


def _numpy_hits(y, sr, fps, n_frames, strength):
    samples_per_frame = sr / fps
    energy = np.zeros(n_frames)
    for i in range(n_frames):
        a, b = int(i * samples_per_frame), int((i + 1) * samples_per_frame)
        chunk = y[a:b]
        energy[i] = np.sqrt(np.mean(chunk ** 2)) if len(chunk) else 0
    flux = np.maximum(0, np.diff(energy, prepend=energy[0]))
    thresh = flux.mean() + strength * flux.std()
    hits = {i for i in range(1, n_frames - 1)
            if flux[i] > thresh and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]}
    energy = energy / (energy.max() + 1e-8)
    return hits, energy


def _load_wav_numpy(path):
    import wave
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels()).mean(axis=1)
    return data / 32768.0, sr


def fake_hits(n_frames, fps, bpm=120):
    """No audio? Invent a steady grid so the edit still has rhythm."""
    step = max(1, int(round(fps * 60 / bpm)))
    hits = set(range(0, n_frames, step))
    energy = np.full(n_frames, 0.4)
    for h in hits:
        energy[h] = 1.0
    return hits, energy
