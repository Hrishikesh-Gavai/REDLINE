"""Neon palette helpers. Config colors are hex strings ("#FF00C8") — easier to tweak than BGR tuples."""


def hex_to_bgr(h):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


class Palette:
    def __init__(self, hex_colors):
        self.colors = [hex_to_bgr(c) for c in hex_colors]

    def at(self, k):
        """k-th color, wrapping around."""
        return self.colors[int(k) % len(self.colors)]

    def __len__(self):
        return len(self.colors)
