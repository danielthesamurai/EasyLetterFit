"""Manual pixel repairs: the clone-stamp brush and the layer it paints into.

This is the answer to everything the automatic rebuild refuses. Where the
background is a face or a hand, no algorithm continues it convincingly, but a
person can see what belongs there in a second and copy it across.

The layer holds colour plus an alpha channel. It was greyscale once, on the
grounds that the pages were black and white line art -- which was true until a
colour page turned up, and then sampling a blue arrow painted grey. The stamp
copies pixels a person pointed at, so it has to keep what they saw. Nothing is
ever written back to the source image.

Undo works on 64px tiles. A full copy of a 600 dpi layer is about 13 MB, which is
far too much to keep per stroke, but a stroke only touches a handful of tiles.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtGui import QImage

TILE = 64

# Tiles a stroke touched, as {(tile_y, tile_x): (value, alpha)}.
TileBackup = dict


class RepairLayer:
    """Painted-over pixels for one page, composited above the artwork."""

    def __init__(self, page: np.ndarray):
        if page.ndim == 2:
            page = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
        self.page = np.ascontiguousarray(page[:, :, :3])
        # Screentone snapping measures dot density, which is a question about
        # light and dark rather than colour. Kept here so it is computed once.
        self.grey = cv2.cvtColor(self.page, cv2.COLOR_BGR2GRAY)
        height, width = self.page.shape[:2]
        self.value = np.zeros((height, width, 3), np.uint8)
        self.alpha = np.zeros((height, width), np.uint8)
        self.dirty: tuple[int, int, int, int] | None = None
        self._brushes: dict[tuple[int, float], np.ndarray] = {}

    # -- state ----------------------------------------------------------

    @property
    def is_empty(self) -> bool:
        return self.dirty is None

    def _mark_dirty(self, x0: int, y0: int, x1: int, y1: int) -> None:
        if self.dirty is None:
            self.dirty = (x0, y0, x1, y1)
            return
        dx0, dy0, dx1, dy1 = self.dirty
        self.dirty = (min(dx0, x0), min(dy0, y0), max(dx1, x1), max(dy1, y1))

    def composite(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        """The page as it currently reads: painted pixels where they exist."""
        alpha = self.alpha[y0:y1, x0:x1, None]
        return np.where(alpha > 0, self.value[y0:y1, x0:x1], self.page[y0:y1, x0:x1])

    # -- brush ----------------------------------------------------------

    def brush(self, radius: int, softness: float) -> np.ndarray:
        """Radial falloff for a dab, cached per size."""
        key = (radius, round(softness, 2))
        cached = self._brushes.get(key)
        if cached is not None:
            return cached

        span = np.arange(-radius, radius + 1, dtype=np.float32)
        distance = np.hypot(span[:, None], span[None, :])
        edge = max(1.0, radius * max(0.01, softness))
        falloff = np.clip((radius - distance) / edge, 0.0, 1.0)

        self._brushes[key] = falloff
        return falloff

    # -- painting -------------------------------------------------------

    def dab(
        self,
        cx: int,
        cy: int,
        sx: int,
        sy: int,
        radius: int,
        softness: float = 0.4,
        backup: TileBackup | None = None,
    ) -> bool:
        """Copy one brush-sized circle from (sx, sy) to (cx, cy)."""
        height, width = self.value.shape[:2]
        x0, y0 = max(0, cx - radius), max(0, cy - radius)
        x1, y1 = min(width, cx + radius + 1), min(height, cy + radius + 1)
        if x1 <= x0 or y1 <= y0:
            return False

        # The source has to be fully on the page, or the copy would smear an
        # edge across the destination.
        offset_x, offset_y = sx - cx, sy - cy
        sx0, sy0 = x0 + offset_x, y0 + offset_y
        sx1, sy1 = x1 + offset_x, y1 + offset_y
        if sx0 < 0 or sy0 < 0 or sx1 > width or sy1 > height:
            return False

        mask = self.brush(radius, softness)
        mask = mask[y0 - (cy - radius) : y0 - (cy - radius) + (y1 - y0),
                    x0 - (cx - radius) : x0 - (cx - radius) + (x1 - x0)]

        if backup is not None:
            self.record(backup, x0, y0, x1, y1)

        source = self.composite(sx0, sy0, sx1, sy1).astype(np.float32)
        target = self.composite(x0, y0, x1, y1).astype(np.float32)
        weight = mask[:, :, None]
        blended = weight * source + (1.0 - weight) * target

        self.value[y0:y1, x0:x1] = blended.astype(np.uint8)
        self.alpha[y0:y1, x0:x1] = np.maximum(
            self.alpha[y0:y1, x0:x1], (mask * 255).astype(np.uint8)
        )
        self._mark_dirty(x0, y0, x1, y1)
        return True

    def stroke(
        self,
        from_point: tuple[int, int],
        to_point: tuple[int, int],
        offset: tuple[int, int],
        radius: int,
        softness: float = 0.4,
        backup: TileBackup | None = None,
    ) -> bool:
        """Dab along a segment, so a fast drag still paints a continuous line."""
        x0, y0 = from_point
        x1, y1 = to_point
        distance = max(abs(x1 - x0), abs(y1 - y0))
        steps = max(1, int(distance / max(1.0, radius * 0.25)))

        painted = False
        for step in range(steps + 1):
            t = step / steps
            px = int(round(x0 + (x1 - x0) * t))
            py = int(round(y0 + (y1 - y0) * t))
            if self.dab(px, py, px + offset[0], py + offset[1], radius, softness, backup):
                painted = True
        return painted

    def erase(
        self,
        cx: int,
        cy: int,
        radius: int,
        backup: TileBackup | None = None,
    ) -> bool:
        """Lift painted pixels back off, revealing the original artwork."""
        height, width = self.value.shape[:2]
        x0, y0 = max(0, cx - radius), max(0, cy - radius)
        x1, y1 = min(width, cx + radius + 1), min(height, cy + radius + 1)
        if x1 <= x0 or y1 <= y0:
            return False

        if backup is not None:
            self.record(backup, x0, y0, x1, y1)

        mask = self.brush(radius, 0.4)
        mask = mask[y0 - (cy - radius) : y0 - (cy - radius) + (y1 - y0),
                    x0 - (cx - radius) : x0 - (cx - radius) + (x1 - x0)]
        remaining = self.alpha[y0:y1, x0:x1].astype(np.float32) * (1.0 - mask)
        self.alpha[y0:y1, x0:x1] = remaining.astype(np.uint8)
        self._mark_dirty(x0, y0, x1, y1)
        return True

    # -- undo -----------------------------------------------------------

    def record(self, backup: TileBackup, x0: int, y0: int, x1: int, y1: int) -> None:
        """Stash the tiles a stroke is about to touch, once each."""
        for ty in range(y0 // TILE, (y1 - 1) // TILE + 1):
            for tx in range(x0 // TILE, (x1 - 1) // TILE + 1):
                if (ty, tx) in backup:
                    continue
                ys, xs = ty * TILE, tx * TILE
                ye = min(ys + TILE, self.value.shape[0])
                xe = min(xs + TILE, self.value.shape[1])
                backup[(ty, tx)] = (
                    self.value[ys:ye, xs:xe].copy(),
                    self.alpha[ys:ye, xs:xe].copy(),
                )

    def swap(self, backup: TileBackup) -> None:
        """Exchange stored tiles with what is there now.

        The same call serves undo and redo, because after swapping, the backup
        holds the state that was just replaced.
        """
        for (ty, tx), (value, alpha) in list(backup.items()):
            ys, xs = ty * TILE, tx * TILE
            ye, xe = ys + value.shape[0], xs + value.shape[1]
            backup[(ty, tx)] = (
                self.value[ys:ye, xs:xe].copy(),
                self.alpha[ys:ye, xs:xe].copy(),
            )
            self.value[ys:ye, xs:xe] = value
            self.alpha[ys:ye, xs:xe] = alpha
            self._mark_dirty(xs, ys, xe, ye)

    # -- output ---------------------------------------------------------

    def region_image(self, x0: int, y0: int, x1: int, y1: int) -> QImage:
        """Premultiplied ARGB of one region, ready to blit."""
        value = self.value[y0:y1, x0:x1].astype(np.uint16)
        alpha = self.alpha[y0:y1, x0:x1]
        premultiplied = ((value * alpha[:, :, None]) // 255).astype(np.uint8)

        height, width = alpha.shape
        argb = np.empty((height, width, 4), np.uint8)
        # Qt's ARGB32 is B, G, R, A in memory on a little-endian machine, which
        # is the order the page is already in.
        argb[:, :, :3] = premultiplied
        argb[:, :, 3] = alpha
        return QImage(argb.data, width, height, 4 * width, QImage.Format_ARGB32_Premultiplied).copy()

    def to_bgra(self) -> np.ndarray:
        """Four-channel array for saving alongside the project."""
        height, width = self.value.shape[:2]
        bgra = np.zeros((height, width, 4), np.uint8)
        bgra[:, :, :3] = self.value
        bgra[:, :, 3] = self.alpha
        return bgra

    def load_bgra(self, bgra: np.ndarray) -> None:
        if bgra.shape[:2] != self.value.shape[:2]:
            return
        self.value = np.ascontiguousarray(bgra[:, :, :3])
        self.alpha = np.ascontiguousarray(bgra[:, :, 3])
        ys, xs = np.nonzero(self.alpha)
        if len(xs):
            self.dirty = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
