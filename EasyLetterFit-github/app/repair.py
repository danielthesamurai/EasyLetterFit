"""Rebuilding the background under text that sits on artwork.

Text inside a balloon is easy: the interior is blank paper, so erasing is a white
fill. Text over a panel is not -- wiping it white punches a hole in the drawing.

What makes this tractable on comic pages is that the grey is not grey. It is a
halftone lattice: a strictly periodic grid of dots. A periodic pattern can be
measured and continued exactly, so instead of inventing pixels this finds a clean
sample of the surrounding tone and tiles it back in on the same phase.

Where the background is not flat and not periodic -- linework, a face, folds of
cloth -- nothing here will guess well, and it says so rather than smearing. That
is what the clone-stamp brush is for.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .imaging import mask_bounds

INK_LEVEL = 128  # below this counts as ink

# A lattice finer than 3px or coarser than 60px is not screentone at 600 dpi.
MIN_PERIOD = 3
MAX_PERIOD = 60

SAMPLE_SIZE = 96  # big enough for several lattice periods, small enough to place
DENSITY_TOLERANCE = 0.035  # how closely a sample must match the surrounding tone
EVENNESS_LIMIT = 0.05  # quadrant spread above this means structure, not tone
PERIODIC_STRENGTH = 0.5  # autocorrelation a real lattice comfortably exceeds


@dataclass
class Repair:
    """A reconstructed patch for one region."""

    patch: np.ndarray  # BGR, sized to the region's bounding box
    origin: tuple[int, int]  # where the patch belongs on the page
    method: str  # "flat" | "tone"
    note: str


def _dominant_period(profile: np.ndarray) -> tuple[int, float] | None:
    """Strongest repeat length in a 1-D ink profile, by autocorrelation."""
    centred = profile.astype(np.float64) - profile.mean()
    if not np.any(centred):
        return None

    correlation = np.correlate(centred, centred, mode="full")[len(centred) - 1 :]
    if correlation[0] <= 0:
        return None
    correlation = correlation / correlation[0]

    window = correlation[MIN_PERIOD : MAX_PERIOD + 1]
    if window.size == 0:
        return None

    best = int(np.argmax(window)) + MIN_PERIOD
    return best, float(correlation[best])


def _lattice_of(patch_ink: np.ndarray) -> tuple[int, int] | None:
    """Lattice periods of a candidate patch, or None if it is not periodic.

    This is the check that separates screentone from artwork. A window of
    linework can match tone on density and still be a drawing; it will not be
    strongly periodic on both axes.
    """
    columns = _dominant_period(patch_ink.sum(axis=0))
    rows = _dominant_period(patch_ink.sum(axis=1))
    if columns is None or rows is None:
        return None
    if columns[1] < PERIODIC_STRENGTH or rows[1] < PERIODIC_STRENGTH:
        return None
    return columns[0], rows[0]


def _window_means(ink: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Mean ink coverage of every window of `size`, via an integral image."""
    integral = cv2.integral(ink.astype(np.uint8))
    h, w = size
    total = integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
    return total.astype(np.float64) / (h * w)


def surrounding_tone(
    gray: np.ndarray, mask: np.ndarray, radius: int, block: int = 16
) -> tuple[float, bool]:
    """Density of the background around a region, and whether it is uniform.

    A plain mean would be dragged upward by any black artwork clipping the ring,
    which then sends the search hunting for an equally dark patch -- that is how
    you end up pasting somebody's sleeve into the hole. Measuring in blocks and
    taking the median ignores the intruder. The agreement figure only says
    whether the ring is readable enough to trust that median; whether the region
    is safe to touch at all is decided by `crossing_structure`.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    ring = cv2.subtract(cv2.dilate(mask, kernel), mask)

    ink = (gray < INK_LEVEL).astype(np.float64)
    ys, xs = np.nonzero(ring)
    if len(xs) == 0:
        return 0.0, False

    densities = []
    for by in range(ys.min(), ys.max() + 1, block):
        for bx in range(xs.min(), xs.max() + 1, block):
            cover = ring[by : by + block, bx : bx + block]
            if cover.size == 0 or (cover > 0).mean() < 0.7:
                continue
            densities.append(float(ink[by : by + block, bx : bx + block].mean()))

    if len(densities) < 4:
        return 0.0, False

    values = np.array(densities)
    median = float(np.median(values))
    agreement = float((np.abs(values - median) <= 0.06).mean())
    return median, agreement >= 0.45


def crossing_structure(
    gray: np.ndarray,
    bbox: tuple[int, int, int, int],
    min_area: int = 120,
    min_inside: int = 40,
    min_outside: int = 150,
) -> bool:
    """Does any drawn shape run through the region and continue beyond it?

    This is what separates lettering from artwork, and density cannot do it.
    Text the author boxed is self-contained: its strokes start and stop inside.
    A jaw, a hair edge or a sleeve carries on past the boundary, so a single
    connected shape has substantial area both inside the box and outside it.
    Paint tone over that and you have cut a rectangle out of the drawing.

    Screentone itself never trips this: its dots are separate specks far below
    `min_area`, even at the densest tone on these pages.
    """
    x, y, w, h = bbox
    page_h, page_w = gray.shape[:2]
    margin = max(40, int(0.3 * max(w, h)))

    wx0, wy0 = max(0, x - margin), max(0, y - margin)
    wx1, wy1 = min(page_w, x + w + margin), min(page_h, y + h + margin)
    window = (gray[wy0:wy1, wx0:wx1] < INK_LEVEL).astype(np.uint8)
    if window.size == 0:
        return False

    count, labels, stats, _ = cv2.connectedComponentsWithStats(window, 8)

    inside = np.zeros(window.shape, dtype=bool)
    inside[max(0, y - wy0) : y - wy0 + h, max(0, x - wx0) : x - wx0 + w] = True

    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_area:
            continue  # a tone dot or a speck, not a structure
        component = labels == index
        within = int(np.count_nonzero(component & inside))
        if within < min_inside:
            continue
        if area - within >= min_outside:
            return True
    return False


def find_tone_samples(
    gray: np.ndarray,
    blocked: tuple[int, int, int, int],
    target_ink: float,
    size: int = SAMPLE_SIZE,
    limit: int = 40,
) -> list[tuple[int, int]]:
    """Candidate clean patches of the same tone, nearest and best-matching first."""
    page_h, page_w = gray.shape[:2]
    if page_h < size or page_w < size:
        return []

    ink = gray < INK_LEVEL
    means = _window_means(ink, (size, size))
    half = size // 2
    quadrant = _window_means(ink, (half, half))

    bx, by, bw, bh = blocked
    margin = 8
    centre = np.array([bx + bw / 2.0, by + bh / 2.0])
    reach = int(3 * max(bw, bh)) + 250

    y_lo, y_hi = max(0, by - reach), min(means.shape[0] - 1, by + bh + reach)
    x_lo, x_hi = max(0, bx - reach), min(means.shape[1] - 1, bx + bw + reach)

    scored: list[tuple[float, tuple[int, int]]] = []
    for sy in range(y_lo, y_hi + 1, 6):
        for sx in range(x_lo, x_hi + 1, 6):
            # Never sample from the region being repaired.
            if (
                sx < bx + bw + margin
                and sx + size > bx - margin
                and sy < by + bh + margin
                and sy + size > by - margin
            ):
                continue

            if abs(means[sy, sx] - target_ink) > DENSITY_TOLERANCE:
                continue

            corners = (
                quadrant[sy, sx],
                quadrant[sy, sx + half],
                quadrant[sy + half, sx],
                quadrant[sy + half, sx + half],
            )
            spread = max(corners) - min(corners)
            if spread > EVENNESS_LIMIT:
                continue

            distance = np.linalg.norm(np.array([sx + half, sy + half]) - centre)
            scored.append(
                (abs(means[sy, sx] - target_ink) * 10 + spread * 5 + distance / 4000.0, (sx, sy))
            )

    scored.sort(key=lambda item: item[0])
    return [position for _, position in scored[:limit]]


def lattice_at(gray: np.ndarray, x: int, y: int, size: int = SAMPLE_SIZE) -> tuple[int, int] | None:
    """Screentone lattice periods around a point, or None if it is not toned."""
    height, width = gray.shape[:2]
    x0 = max(0, min(width - size, x - size // 2))
    y0 = max(0, min(height - size, y - size // 2))
    if width < size or height < size:
        return None
    patch = (gray[y0 : y0 + size, x0 : x0 + size] < INK_LEVEL).astype(np.float64)
    return _lattice_of(patch)


def _flat_patch(colour: np.ndarray, mask: np.ndarray, box, dark: bool) -> np.ndarray:
    """A patch of the one colour the surroundings are made of.

    Flat does not mean white. A caption laid over a coloured panel has a flat
    background that happens to be blue, and filling it with paper white would be
    every bit as wrong as the grey this used to produce.
    """
    x, y, w, h = box
    ring = cv2.subtract(
        cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))), mask
    )
    pixels = colour[ring > 0]
    if pixels.size == 0:
        shade = 0 if dark else 255
        return np.full((h, w, 3), shade, np.uint8)
    tone = np.median(pixels.reshape(-1, 3), axis=0).astype(np.uint8)
    return np.repeat(np.repeat(tone[None, None, :], h, axis=0), w, axis=1)


def repair_region(
    gray: np.ndarray, mask: np.ndarray, colour: np.ndarray | None = None
) -> Repair | None:
    """Rebuild whatever the background is under `mask`.

    Every decision -- is this flat, is it periodic, is it safe to touch -- is
    made on the grey page, because they are all questions about light and dark.
    The pixels handed back are taken from the colour page, because they are what
    ends up on the page.

    Returns None when the surroundings are too detailed to continue safely --
    better to leave the artwork alone and say so than to smear it.
    """
    if colour is None:
        colour = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    x, y, w, h = mask_bounds(mask)
    if w == 0 or h == 0:
        return None

    # Never paint over a shape that carries on outside the box.
    if crossing_structure(gray, (x, y, w, h)):
        return None

    radius = max(10, int(0.3 * max(w, h)))
    density, coherent = surrounding_tone(gray, mask, radius)
    if not coherent:
        return None  # surroundings too mixed to read a background from

    if density <= 0.02:
        patch = _flat_patch(colour, mask, (x, y, w, h), dark=False)
        shade = tuple(int(v) for v in patch[0, 0])
        return Repair(patch, (x, y), "flat", f"flat fill, BGR {shade}")
    if density >= 0.98:
        patch = _flat_patch(colour, mask, (x, y, w, h), dark=True)
        shade = tuple(int(v) for v in patch[0, 0])
        return Repair(patch, (x, y), "flat", f"flat fill, BGR {shade}")

    for sx, sy in find_tone_samples(gray, (x, y, w, h), density):
        sample = (gray[sy : sy + SAMPLE_SIZE, sx : sx + SAMPLE_SIZE] < INK_LEVEL).astype(
            np.float64
        )
        lattice = _lattice_of(sample)
        if lattice is None:
            continue
        period_x, period_y = lattice

        # Tile using whole numbers of periods, so the dot grid in the patch stays
        # in step with the grid around it instead of visibly jumping.
        tile_w = (SAMPLE_SIZE // period_x) * period_x
        tile_h = (SAMPLE_SIZE // period_y) * period_y
        if tile_w < period_x or tile_h < period_y:
            continue

        xs = sx + np.mod(np.arange(x, x + w) - sx, tile_w)
        ys = sy + np.mod(np.arange(y, y + h) - sy, tile_h)
        patch = colour[np.ix_(ys, xs)].copy()

        return Repair(
            patch, (x, y), "tone", f"screentone rebuilt, {period_x}x{period_y} px lattice"
        )

    return None
