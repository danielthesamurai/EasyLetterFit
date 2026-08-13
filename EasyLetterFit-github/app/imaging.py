"""Image analysis for comic pages: binarisation, bubble detection, background repair.

Everything here works on the *pristine* page pixels. Edits are never written back
into the source image; the editor composites text and repairs on top.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

# Pixels at or above this grey level count as blank paper. The pages are
# effectively 1-bit with four levels of antialiasing (0/64/128/191/255), so a
# high threshold keeps the AA fringe on the "ink" side and stops flood fills
# seeping through balloon outlines.
PAPER_THRESHOLD = 200


def imread_unicode(path: str, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    """Read an image whose path may contain non-ASCII characters.

    cv2.imread cannot open these on Windows -- it goes through a narrow-char
    file API, so a Japanese folder or filename simply returns None. Reading the
    bytes in Python and decoding them sidesteps the filename entirely.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: str, image: np.ndarray) -> bool:
    """Write an image to a path that may contain non-ASCII characters."""
    suffix = os.path.splitext(path)[1] or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False
    try:
        encoded.tofile(path)
    except OSError:
        return False
    return True


def load_colour(path: str) -> np.ndarray:
    """Load a page as three-channel BGR, flattening any alpha onto white."""
    raw = imread_unicode(path)
    if raw is None:
        raise OSError(f"could not read image: {path}")

    if raw.ndim == 3 and raw.shape[2] == 4:
        rgb = raw[:, :, :3].astype(np.float32)
        alpha = raw[:, :, 3:4].astype(np.float32) / 255.0
        raw = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)

    if raw.ndim == 2:
        return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(raw[:, :, :3])


def load_gray(path: str) -> np.ndarray:
    """Load a page as a single-channel greyscale array, flattening alpha onto white.

    Detection and the tone rebuild read shape and density, not hue, so they work
    on this. The clone stamp does not -- it copies pixels a person picked, and
    those have colours.
    """
    return cv2.cvtColor(load_colour(path), cv2.COLOR_BGR2GRAY)


def page_threshold(gray: np.ndarray) -> int:
    """Where paper ends and ink begins, for this particular page.

    A fixed level works on clean exports, where the art is effectively 1-bit.
    It does not work on a scan: the paper is grey rather than white, the
    outlines are softened by compression, and a level tuned for one puts holes
    in the other\'s balloon outlines -- which is how a flood escapes a balloon
    and swallows the panel. Otsu adapts, clamped so it cannot do anything wild.
    """
    # Anchor to where this page's paper actually sits, then come down by the
    # same margin the fixed level used. Otsu is wrong here: it lands mid-way
    # between ink and paper, which puts antialiasing on the paper side and lets
    # floods seep straight through a balloon outline.
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    paper = int(np.argmax(histogram[128:])) + 128
    return int(np.clip(paper - 55, 100, PAPER_THRESHOLD))


def paper_mask(gray: np.ndarray, level: int | None = None) -> np.ndarray:
    """255 where the page is blank paper, 0 where there is ink."""
    if level is None:
        level = PAPER_THRESHOLD
    return ((gray >= level).astype(np.uint8)) * 255


def fill_mask(gray: np.ndarray, dark: bool, level: int | None = None) -> np.ndarray:
    """255 where a balloon of the given polarity has its fill."""
    if level is None:
        level = PAPER_THRESHOLD
    if dark:
        return ((gray < level).astype(np.uint8)) * 255
    return paper_mask(gray, level)


# Scanned outlines are rarely unbroken, and one pixel of daylight is enough for
# a flood to pour out of a balloon. Sealing is tried gently first and firmly
# second: the wider kernel closes gaps up to about four pixels, but takes two
# pixels off the fill, which is enough to destroy a small balloon on a
# low-resolution page. Gentle-then-firm gets both.
_SEALS = [
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
]


def sealed_fill(gray: np.ndarray, dark: bool, level: int | None = None):
    """The fill mask, with hairline gaps in outlines closed.

    A scanned balloon outline is rarely unbroken. One pixel of daylight is
    enough for a flood to pour out of the balloon and across the panel, and on a
    toned page the tone\'s own gaps then connect to everything. Eroding before
    the flood seals those; the region is grown back afterwards so the balloon
    keeps its true edge.
    """
    field = fill_mask(gray, dark, level)
    return field, [cv2.erode(field, seal) for seal in _SEALS]


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Close interior holes, so a balloon's white interior swallows its lettering."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled



def _seed_candidates(
    field: np.ndarray, x: int, y: int, radius: int, limit: int = 3
) -> list[tuple[int, int]]:
    """Places near the click worth flooding from, biggest first.

    Taking simply the nearest matching pixel fails badly on a busy page. Click
    just outside a black balloon on a panel full of speed lines and the closest
    black thing is a line, which floods to a sliver and gets rejected -- while
    the balloon sits a few pixels further away, untried. Ranking the distinct
    shapes around the click by how much of the neighbourhood they occupy puts
    the balloon first, and the passing line last.
    """
    h, w = field.shape[:2]
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    window = field[y0:y1, x0:x1]
    if window.size == 0 or not window.any():
        return []

    count, labels, stats, _ = cv2.connectedComponentsWithStats((window > 0).astype(np.uint8), 8)
    if count <= 1:
        return []

    here = labels[y - y0, x - x0] if field[y, x] else 0
    order = sorted(range(1, count), key=lambda i: -int(stats[i, cv2.CC_STAT_AREA]))
    if here:
        # Whatever the click actually landed on is tried first regardless.
        order.remove(here)
        order.insert(0, here)

    seeds: list[tuple[int, int]] = []
    for label in order[:limit]:
        ys, xs = np.nonzero(labels == label)
        dy = ys + y0 - y
        dx = xs + x0 - x
        i = int(np.argmin(dx * dx + dy * dy))
        seeds.append((int(xs[i] + x0), int(ys[i] + y0)))
    return seeds


def _search_radius(gray: np.ndarray) -> int:
    """How far from the click to look for something to flood."""
    h, w = gray.shape[:2]
    return max(60, min(h, w) // 25)


def _solidity(mask: np.ndarray) -> float:
    """How close the region is to its own convex hull.

    A balloon interior is a blob -- an ellipse, a rounded box, a spiky shout --
    and nearly fills its hull. A panel background is not: it wraps around
    figures and props, so its hull is far bigger than it is.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    hull_area = cv2.contourArea(cv2.convexHull(largest))
    if hull_area <= 0:
        return 0.0
    return float(cv2.contourArea(largest) / hull_area)




# How balloon-like each measurement is, and how much it counts. Weighted
# evidence rather than a chain of pass/fail gates: with hard gates, every
# threshold has to be loose enough to admit the most extreme real balloon -- a
# spiky shout scores worse on form than some panel backgrounds do -- and that
# looseness is then granted to everything else. Scoring lets a candidate that is
# mildly wrong on several counts fail, which is what artwork looks like, while a
# balloon that is poor on one and excellent on the rest still passes.
FEATURE_RAMPS = {
    # name:        (low, high, weight) -- more is better, saturating at high
    "holes": (8, 25, 0.8),  # a balloon holds many separate marks
    "solidity": (0.45, 0.90, 0.5),  # blob-ish, but shouts are spiky
    "uniformity": (0.25, 0.55, 0.5),  # letters are of a size; faces are not
}

# Typical mark size, relative to the region. Also a band: too small is
# screentone, and too large is not lettering at all. Flooding a panel's white
# background gives a region whose holes are the artwork -- a figure, a balloon,
# a chunk of black -- and rewarding size without limit hands that full marks.
# A letter is a couple of percent of its balloon; these are tens of percent.
COARSENESS_BAND = (0.0005, 0.0015, 0.020, 0.050)
COARSENESS_WEIGHT = 1.0

# How much of the region is lettering. A band, not a ramp: too little means
# empty artwork, and too much means the region is mostly holes, which is not a
# balloon interior either. Left as an ever-rising score, a shape that is 90%
# holes earns full marks for it.
INK_BAND = (0.015, 0.06, 0.40, 0.60)
INK_WEIGHT = 1.2

MIN_BALLOON_SCORE = 2.3  # of a possible 4.0

# Score at which a candidate is convincing enough to stop looking. Weighing
# every polarity, sealing and seed costs about a dozen full-page flood fills;
# a real balloon usually scores well above this on the first one tried.
GOOD_ENOUGH_SCORE = 3.4

# Regions bigger than this are measured at reduced resolution. Every feature is
# a ratio, so they survive scaling, and the regions this affects are the sprawls
# that get rejected anyway -- there is no sense tracing contours across a
# megapixel to decide that a panel is not a balloon.
ANALYSIS_LIMIT = 700 * 700

# Hard gates, for what no score should be able to argue with.
MAX_AREA_FRACTION = 0.35  # bigger than this is a panel, not a balloon

# A balloon holds several separate marks. Below that, two of the measurements
# stop meaning anything -- with a single hole, "uniformity" is 1.0 by definition
# and "coarseness" is enormous -- so a stray shape with one detail inside scores
# better than a real balloon. Every balloon measured had at least 23.
MIN_HOLES = 4

# A balloon has lettering in it -- that is why you clicked it. A region that is
# essentially empty is background, however well shaped it may be.
MIN_INK = 0.02

# And the mirror case: if most of the enclosed area is not fill, the fill is a
# thin frame around something -- a panel border -- rather than a balloon body.
MAX_INK = 0.60

# The marks inside must be of a size with each other. This is what says the
# holes are all lettering rather than lettering plus a great chunk of artwork:
# flooding a balloon fused with a panel gives one enormous hole among the
# letters, and the ratio collapses. Real balloons measured 0.37 to 0.74; a
# fused sprawl measures 0.02, and a patch of shaded artwork 0.31.
MIN_UNIFORMITY = 0.33

# -- merged balloons ----------------------------------------------------
# Two balloons drawn overlapping are one shape: each outline is drawn only
# outside the other, so no line divides them and a flood fills both. They are
# still two balloons, and clicking one has to give you that one.
#
# The join always leaves a waist. Eroding the region pinches it apart there
# first, so the question "is this two balloons?" is answered by "does it come
# apart into two substantial pieces before either piece disappears?".

# At the radius where the pinch first opens, the surviving cores are heavily
# eroded and small -- so their area says little. This floor only exists to
# ignore slivers.
MIN_LOBE_FRACTION = 0.02

# What actually tells a second balloon from a tail or a bump on the outline is
# how thick it is in its own right. A balloon is within reach of the widest part
# of the region; a nub is nowhere near. Measured pairs came apart into lobes
# whose thinner half still reached 0.75-0.98 of the region's widest point, so
# there is a wide margin here.
MIN_LOBE_REACH = 0.30

# How finely to look for the radius that separates them. The cut lands on white
# paper between two balloons, so a coarse cut is invisible -- this only needs to
# be fine enough to find the pinch, not to place it exactly.
SPLIT_STEPS = 24

# Splitting is measured on a reduced copy: the waist is tens of pixels across on
# a page this size, far coarser than the resolution it is being found at.
SPLIT_LIMIT = 260 * 260


def _split_at_waist(region: np.ndarray, click: tuple[int, int]):
    """Yield the balloon under the click when a region is several joined ones.

    Yields nothing when the region does not come apart, which is the common
    case and costs one distance transform on a thumbnail.
    """
    bx, by, bw, bh = cv2.boundingRect(region)
    cx, cy = click
    if not (bx <= cx < bx + bw and by <= cy < by + bh):
        return

    filled = _fill_holes(region[by : by + bh, bx : bx + bw])

    shrink = 1.0
    work = filled
    if filled.size > SPLIT_LIMIT:
        shrink = (SPLIT_LIMIT / filled.size) ** 0.5
        work = cv2.resize(
            filled,
            (max(8, int(bw * shrink)), max(8, int(bh * shrink))),
            interpolation=cv2.INTER_NEAREST,
        )
    # distanceTransform measures to the nearest background *pixel*, so a mask
    # that runs to the edge of its own bounding box reads as far thicker than it
    # is. One ring of background restores the truth.
    work = cv2.copyMakeBorder(work, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    distance = cv2.distanceTransform((work > 0).astype(np.uint8), cv2.DIST_L2, 5)
    peak = float(distance.max())
    if peak < 6.0:
        return

    area = int(cv2.countNonZero(work))
    minimum_lobe = MIN_LOBE_FRACTION * area

    # Erode by raising a threshold on the distance transform -- the same shape
    # an erosion would give, without running one per radius.
    cores = None
    for step in range(1, SPLIT_STEPS):
        radius = peak * step / SPLIT_STEPS
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (distance > radius).astype(np.uint8), 8
        )
        lobes = [
            index
            for index in range(1, count)
            if stats[index, cv2.CC_STAT_AREA] >= minimum_lobe
        ]
        if len(lobes) < 2:
            continue

        # Two pieces is not enough on its own -- a bump on a balloon's outline
        # also parts from it. Both have to be balloon-thick.
        thick = [
            index
            for index in lobes
            if float(distance[labels == index].max()) >= MIN_LOBE_REACH * peak
        ]
        if len(thick) < 2:
            continue
        cores = (labels, thick)
        break

    if cores is None:
        return  # one balloon, however lumpy

    labels, lobes = cores
    # Hand every pixel to the nearest core. For balloons joined at a waist that
    # boundary is the waist itself, which is where the artist drew the join.
    seeds = np.full(work.shape, 255, np.uint8)
    for index in lobes:
        seeds[labels == index] = 0
    _, ownership = cv2.distanceTransformWithLabels(
        seeds, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_CCOMP
    )

    wx = min(work.shape[1] - 2, max(1, int((cx - bx) * shrink) + 1))
    wy = min(work.shape[0] - 2, max(1, int((cy - by) * shrink) + 1))
    mine = ownership[wy, wx]
    piece = ((ownership == mine).astype(np.uint8) * 255)[1:-1, 1:-1]

    if shrink != 1.0:
        piece = cv2.resize(piece, (bw, bh), interpolation=cv2.INTER_NEAREST)

    whole = np.zeros_like(region)
    whole[by : by + bh, bx : bx + bw] = cv2.bitwise_and(piece, filled)
    yield cv2.bitwise_and(region, whole)


# -- deeply overlapped balloons -----------------------------------------
# When one balloon is drawn well inside another there is no waist to pinch:
# eroding just walks in towards the bigger one's middle and the smaller never
# becomes a piece of its own. What is still there is the pair of corners where
# the two outlines cross, one on each side. Cutting from one to the other is the
# join the artist drew.

# A notch has to be a real corner, not a wobble in the ink. Measured as a
# fraction of the region's smaller side.
MIN_NOTCH_DEPTH = 0.09

# Both halves have to come out looking like balloon bodies. This is what keeps
# a shout balloon intact: cutting between two of its spikes leaves two pieces
# that are still spiky, while cutting two overlapped ovals apart leaves two
# ovals. Real balloons measure 0.86 and up.
MIN_PIECE_SOLIDITY = 0.88


def _split_at_notches(region: np.ndarray, click: tuple[int, int]):
    """Yield the balloon under the click by cutting corner to corner."""
    bx, by, bw, bh = cv2.boundingRect(region)
    cx, cy = click
    if not (bx <= cx < bx + bw and by <= cy < by + bh):
        return

    filled = _fill_holes(region[by : by + bh, bx : bx + bw])

    shrink = 1.0
    work = filled
    if filled.size > SPLIT_LIMIT:
        shrink = (SPLIT_LIMIT / filled.size) ** 0.5
        work = cv2.resize(
            filled,
            (max(8, int(bw * shrink)), max(8, int(bh * shrink))),
            interpolation=cv2.INTER_NEAREST,
        )

    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return
    outline = max(contours, key=cv2.contourArea)
    if len(outline) < 8:
        return

    try:
        hull = cv2.convexHull(outline, returnPoints=False)
        if hull is None or len(hull) < 4:
            return
        defects = cv2.convexityDefects(outline, hull)
    except cv2.error:
        return  # degenerate outline; nothing to read
    if defects is None or len(defects) < 2:
        return
    defects = defects.reshape(-1, 4)

    floor = MIN_NOTCH_DEPTH * min(work.shape[:2])
    # convexityDefects reports depth in fixed point, 8 fractional bits.
    notches = sorted(
        (
            (float(depth) / 256.0, tuple(int(v) for v in outline[far][0]))
            for _, _, far, depth in defects
            if float(depth) / 256.0 >= floor
        ),
        reverse=True,
    )
    if len(notches) < 2:
        return

    area = int(cv2.countNonZero(work))
    click_x = min(work.shape[1] - 1, max(0, int((cx - bx) * shrink)))
    click_y = min(work.shape[0] - 1, max(0, int((cy - by) * shrink)))

    # Try the deepest notches against each other. The pair that is really the
    # join gives two balloon-shaped halves; any other pair does not.
    for first in range(min(3, len(notches))):
        for second in range(first + 1, min(4, len(notches))):
            cut = work.copy()
            cv2.line(cut, notches[first][1], notches[second][1], 0, 3)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(cut, 8)
            if count != 3:  # background plus exactly two halves
                continue
            if any(
                stats[index, cv2.CC_STAT_AREA] < MIN_LOBE_FRACTION * area
                for index in (1, 2)
            ):
                continue
            if any(
                _solidity(((labels == index).astype(np.uint8)) * 255) < MIN_PIECE_SOLIDITY
                for index in (1, 2)
            ):
                continue

            mine = labels[click_y, click_x]
            if mine == 0:
                continue
            piece = ((labels == mine).astype(np.uint8)) * 255
            if shrink != 1.0:
                piece = cv2.resize(piece, (bw, bh), interpolation=cv2.INTER_NEAREST)

            whole = np.zeros_like(region)
            whole[by : by + bh, bx : bx + bw] = cv2.bitwise_and(piece, filled)
            yield cv2.bitwise_and(region, whole)
            return


# -- two blocks of lettering in one balloon ------------------------------
# A balloon can hold two separate blocks with a deliberate gap between them --
# a pause, an afterthought, a second sentence set apart. The shape says nothing
# about it; only the writing does. Blocks want their own boxes, because they
# want their own placement, which is what a letterer gives them by hand.
#
# Found by smearing the original lettering until glyphs of the same block run
# together and blocks do not. The smear is measured in characters, so it works
# at any page size and in either writing direction.

# Smear radius as a fraction of a character. Measured across 41 balloons a
# person had lettered: every one stayed a single block from 0.5 upward, and the
# balloon that really does hold two came apart anywhere below 0.8. At 0.4 a
# single block started falling apart into its own lines.
BLOCK_GAP = 0.6

# A block worth its own box. Below this it is a stray mark, a tail of furigana,
# or a piece of the balloon's own outline that survived.
MIN_BLOCK_SHARE = 0.12


def _split_at_blocks(gray: np.ndarray, region: np.ndarray, click: tuple[int, int]):
    """Yield the block of lettering under the click, when a region holds several."""
    bx, by, bw, bh = cv2.boundingRect(region)
    cx, cy = click
    if not (bx <= cx < bx + bw and by <= cy < by + bh):
        return

    filled = _fill_holes(region[by : by + bh, bx : bx + bw])
    patch = gray[by : by + bh, bx : bx + bw]

    shrink = 1.0
    if filled.size > SPLIT_LIMIT:
        shrink = (SPLIT_LIMIT / filled.size) ** 0.5
        size = (max(8, int(bw * shrink)), max(8, int(bh * shrink)))
        filled = cv2.resize(filled, size, interpolation=cv2.INTER_NEAREST)
        patch = cv2.resize(patch, size, interpolation=cv2.INTER_AREA)

    # The lettering, kept clear of the balloon's own outline.
    edge = max(2, int(round(4 * shrink)))
    inner = cv2.erode(
        filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge * 2 + 1,) * 2)
    )
    ink = cv2.bitwise_and(
        (patch < page_threshold(gray)).astype(np.uint8) * 255, inner
    )
    if cv2.countNonZero(ink) == 0:
        return

    # How big is a character here? The marks are glyphs and strokes, so their
    # typical extent is the scale everything else should be measured against.
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    if count <= 2:
        return
    extents = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT])
    character = float(np.median(extents[extents > 2])) if (extents > 2).any() else 0.0
    if character < 3:
        return

    radius = max(1, int(round(character * BLOCK_GAP)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    smear = cv2.dilate(ink, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(smear, 8)
    total = float(cv2.countNonZero(ink))
    blocks = [
        index
        for index in range(1, n)
        if cv2.countNonZero(cv2.bitwise_and(((labels == index).astype(np.uint8) * 255), ink))
        >= MIN_BLOCK_SHARE * total
    ]
    if len(blocks) < 2:
        return

    # Hand every pixel of the region to the nearest block, the same way joined
    # balloons are divided at their waist. The boundary falls in the blank paper
    # between the blocks, so both halves still erase cleanly.
    seeds = np.full(ink.shape, 255, np.uint8)
    for index in blocks:
        seeds[labels == index] = 0
    _, ownership = cv2.distanceTransformWithLabels(
        seeds, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_CCOMP
    )

    wx = min(ink.shape[1] - 1, max(0, int((cx - bx) * shrink)))
    wy = min(ink.shape[0] - 1, max(0, int((cy - by) * shrink)))
    piece = ((ownership == ownership[wy, wx]).astype(np.uint8) * 255)

    if shrink != 1.0:
        piece = cv2.resize(piece, (bw, bh), interpolation=cv2.INTER_NEAREST)

    whole = np.zeros_like(region)
    whole[by : by + bh, bx : bx + bw] = piece
    yield cv2.bitwise_and(region, whole)


def _split_balloons(
    gray: np.ndarray, region: np.ndarray, click: tuple[int, int], accepted: bool
):
    """The piece of a region under the click, when it holds more than one thing.

    Two ways balloons get joined, needing two different reads. Overlap them a
    little and the join leaves a waist, which pinching finds. Overlap them a lot
    and there is no waist left -- but the corners where the outlines cross are
    still there, and the cut between them is the join. A third case is one
    balloon holding two separate blocks of lettering, where nothing about the
    shape says so and only the writing inside does.

    The two are not equally strong evidence, so they are not trusted equally.
    Coming apart at a waist says on its own that there were two bodies here, and
    is allowed to find a balloon in a region that was going to be turned down. A
    pair of facing corners is much weaker -- a chin and a fringe make two, and
    cutting between them leaves a perfectly convincing face -- so that one may
    only narrow a region already accepted as a balloon, never promote one.
    """
    for piece in _split_at_waist(region, click):
        yield piece
        return
    if not accepted:
        return
    for piece in _split_at_notches(region, click):
        yield piece
        return
    yield from _split_at_blocks(gray, region, click)


def _ramp(value: float, low: float, high: float) -> float:
    if high <= low:
        return 1.0 if value >= high else 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _region_features(region: np.ndarray, filled: np.ndarray, page_area: int) -> dict | None:
    """Measure a flooded region: what it is made of, not how big it is."""
    if region.size > ANALYSIS_LIMIT:
        scale = (ANALYSIS_LIMIT / region.size) ** 0.5
        size = (max(8, int(region.shape[1] * scale)), max(8, int(region.shape[0] * scale)))
        region = cv2.resize(region, size, interpolation=cv2.INTER_NEAREST)
        filled = cv2.resize(filled, size, interpolation=cv2.INTER_NEAREST)

    area = int(np.count_nonzero(region))
    filled_area = int(np.count_nonzero(filled))
    if filled_area == 0:
        return None

    holes = cv2.subtract(filled, region)
    count, _, stats, _ = cv2.connectedComponentsWithStats((holes > 0).astype(np.uint8), 8)
    if count <= 1:
        return None

    sizes = np.sort(stats[1:, cv2.CC_STAT_AREA])[::-1]

    # Drop specks before measuring anything. On a scan the fill is peppered with
    # thousands of noise holes a few pixels across, and they swamp the median:
    # a real balloon came back reporting 2161 holes whose typical size was
    # 0.0004% of it, when the dozen actual letters were hundreds of times that.
    # Clean digital art has no specks, so this changes nothing there.
    cutoff = max(8.0, 0.0005 * filled_area)
    marks = sizes[sizes >= cutoff]
    if marks.size == 0:
        return None

    # Judge evenness on the larger marks.
    larger = marks[: max(1, len(marks) // 2)]
    biggest = float(larger.max())

    return {
        "area": filled_area / page_area,
        "solidity": _solidity(filled),
        "holes": int(marks.size),
        "coarseness": float(np.median(marks)) / filled_area,
        "uniformity": float(np.median(larger)) / biggest if biggest else 0.0,
        # Lettering only. Everything-that-is-not-fill would also count every
        # noise speck on a scanned page, which inflates it enormously -- a real
        # balloon reported 0.39 where clean art never exceeds 0.17 -- and then
        # trips a ceiling calibrated on clean art. On clean art the two are the
        # same number, because there are no specks to leave out.
        "ink": float(marks.sum()) / filled_area,
    }


def _band(value: float, rise_from: float, rise_to: float, fall_from: float, fall_to: float) -> float:
    """1.0 across the middle, tapering to 0 outside on both sides."""
    if value <= rise_to:
        return _ramp(value, rise_from, rise_to)
    return 1.0 - _ramp(value, fall_from, fall_to)


def _balloon_score(features: dict) -> float:
    """Total weighted evidence that a region is a speech balloon."""
    score = INK_WEIGHT * _band(features["ink"], *INK_BAND)
    score += COARSENESS_WEIGHT * _band(features["coarseness"], *COARSENESS_BAND)
    return score + sum(
        _ramp(features[name], low, high) * weight
        for name, (low, high, weight) in FEATURE_RAMPS.items()
    )


def detect_bubble(
    gray: np.ndarray,
    x: int,
    y: int,
    max_area_fraction: float = MAX_AREA_FRACTION,
    min_area_px: int = 400,
    min_score: float = MIN_BALLOON_SCORE,
    min_holes: int = MIN_HOLES,
    min_ink: float = MIN_INK,
    max_ink: float = MAX_INK,
    min_uniformity: float = MIN_UNIFORMITY,
) -> np.ndarray | None:
    """Find the speech-balloon interior containing (x, y).

    Floods the blank paper around the click until it meets the balloon outline,
    then fills the letter-shaped holes to recover the solid interior. Returns a
    (mask, dark) where the mask is 255 inside, and `dark` says the balloon is
    filled black with light lettering. None if this is not a balloon.

    Three guards separate balloons from ordinary page background:

    - **Hole size.** The holes in a balloon are its letters; the holes in flooded
      screentone are its dots. Comparing the median hole against the region
      containing it tells those apart by an order of magnitude, and is the guard
      that rejects tone.
    - **Shape.** A balloon is roughly a blob; panel background wraps around
      whatever is drawn in it. The bar has to be low -- a spiky shout balloon
      scores worse than some panel backgrounds, because its points push the
      convex hull far out -- so this only catches the badly straggling cases.
      Unlike an area threshold it does not care how big the page is.
    - **Ink inside.** A balloon holds lettering; empty panel background holds
      almost nothing.
    - **Area**, only as a loose backstop. It has to be generous: a black balloon
      merged with black hair, or a big balloon on a small page, are both far
      more of the page than a balloon usually is, and shape tells them apart
      from a panel perfectly well.
    """
    h, w = gray.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return None

    # Try the polarity the click landed on first. An inverted balloon -- white
    # lettering on a black fill -- is just as much a balloon, but flooding the
    # blank paper from inside one escapes immediately, because the fill itself
    # reads as ink.
    level = page_threshold(gray)
    started_dark = bool(gray[int(y), int(x)] < level)

    # Both polarities are tried and the more convincing one wins. Taking the
    # first that merely passes goes wrong whenever a click lands on a letter
    # stroke rather than the fill around it: the wrong polarity is then tried
    # first, and a mediocre-but-passing region beats the balloon that was
    # sitting there scoring higher.
    best = None
    for dark in (started_dark, not started_dark):
        found = _detect_polarity(
            level,
            gray,
            int(x),
            int(y),
            dark,
            max_area_fraction,
            min_area_px,
            min_score,
            min_holes,
            min_ink,
            max_ink,
            min_uniformity,
        )
        if found is None:
            continue
        if best is None or found[0] > best[0][0]:
            best = (found, dark)
            if found[0] >= GOOD_ENOUGH_SCORE:
                break

    if best is None:
        return None

    (score, (bx, by), crop), dark = best
    mask = np.zeros(gray.shape[:2], np.uint8)
    mask[by : by + crop.shape[0], bx : bx + crop.shape[1]] = crop
    return mask, dark


def _detect_polarity(
    level: int,
    gray: np.ndarray,
    x: int,
    y: int,
    dark: bool,
    max_area_fraction: float,
    min_area_px: int,
    min_score: float,
    min_holes: int,
    min_ink: float,
    max_ink: float,
    min_uniformity: float,
):
    """Best balloon-like region for one polarity, as (score, origin, crop)."""
    h, w = gray.shape[:2]
    field, sealings = sealed_fill(gray, dark, level)

    best = None
    for seal, sealed in zip(_SEALS, sealings):
        for seed in _seed_candidates(sealed, x, y, _search_radius(gray)):
            found = _from_seed(
                gray,
                sealed,
                field,
                seal,
                seed,
                (x, y),
                max_area_fraction,
                min_area_px,
                min_score,
                min_holes,
                min_ink,
                max_ink,
                min_uniformity,
            )
            if found is not None and (best is None or found[0] > best[0]):
                best = found
                if best[0] >= GOOD_ENOUGH_SCORE:
                    return best
    return best


def _from_seed(
    gray: np.ndarray,
    sealed: np.ndarray,
    field: np.ndarray,
    seal: np.ndarray,
    seed: tuple[int, int],
    click: tuple[int, int],
    max_area_fraction: float,
    min_area_px: int,
    min_score: float,
    min_holes: int,
    min_ink: float,
    max_ink: float,
    min_uniformity: float,
):
    """Flood from one seed and score it, as (score, origin, crop)."""
    h, w = gray.shape[:2]

    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(
        sealed.copy(),
        flood_mask,
        seed,
        newVal=128,
        loDiff=0,
        upDiff=0,
        flags=4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8),
    )
    # Sealing decides the region's outline, and must not touch what is inside
    # it. Growing the flood back directly is an opening of the fill, which
    # permanently erases thin structure -- so the many small holes of real
    # lettering merge into a few large even ones, and a patch of artwork starts
    # measuring like a perfect balloon. Take the sealed flood only as an extent,
    # then read the untouched fill within it.
    extent = _fill_holes(cv2.dilate(flood_mask[1:-1, 1:-1], seal))
    region = cv2.bitwise_and(field, extent)

    page_area = h * w

    def judge(candidate: np.ndarray):
        area = int(cv2.countNonZero(candidate))
        if area < min_area_px or area > max_area_fraction * page_area:
            return None  # impossible on size alone; no score can rescue it

        # Everything below is measured on the region's own bounding box. A
        # balloon is a few hundred pixels across on a 6.5 MP page, and tracing
        # contours over the whole sheet to look at it costs an order of
        # magnitude more than the answer is worth.
        bx, by, bw, bh = cv2.boundingRect(candidate)
        crop = candidate[by : by + bh, bx : bx + bw]
        filled_crop = _fill_holes(crop)

        features = _region_features(crop, filled_crop, page_area)
        if features is None or features["holes"] < min_holes:
            return None
        if not (min_ink <= features["ink"] <= max_ink):
            return None
        if features["uniformity"] < min_uniformity:
            return None

        # The balloon has to be the thing under the cursor. Seeds are allowed
        # to walk away from the click -- that is what makes clicking a letter
        # work, since the letter is ink and the fill is around it -- but the
        # region they find must still cover where you actually pointed.
        cx, cy = click
        if not (bx <= cx < bx + bw and by <= cy < by + bh):
            return None
        if not filled_crop[cy - by, cx - bx]:
            return None

        # Hand back the crop and where it goes. Building a page-sized mask for
        # every candidate is wasted work when all but one are discarded.
        return _balloon_score(features), (bx, by), filled_crop

    best = judge(region)

    # Balloons drawn overlapping flood as one shape, and that shape scores like
    # a balloon -- because it is two of them. So this has to be asked before the
    # good-enough shortcut, not after it: the merged reading is never the right
    # answer when the region really is several balloons, however well it scores.
    accepted = best is not None and best[0] >= min_score
    for piece in _split_balloons(gray, region, click, accepted):
        separated = judge(piece)
        if separated is not None:
            return separated

        # One balloon on its own may not pass on its own measurements even
        # though the pair did. Both ways that happens are ordinary lettering:
        # a balloon holding two or three characters has too few marks to look
        # even, and furigana are deliberately a fraction of the size of the
        # glyphs they sit beside, which is exactly what "uneven marks" measures.
        #
        # It does not need to prove itself again. The region it was cut from
        # already did, and cutting a balloon at its waist gives balloons. This
        # can only narrow an answer that was going to be returned anyway --
        # never turn a refusal into a detection.
        if best is not None and best[0] >= min_score:
            inherited = _piece_candidate(piece, best[0], click, min_area_px)
            if inherited is not None:
                return inherited

    if best is not None and best[0] >= GOOD_ENOUGH_SCORE:
        return best

    # A black balloon drawn against black artwork -- brush strokes, speed lines,
    # a panel edge -- floods as one sprawling shape with all of it, and a shape
    # that sprawls is exactly what poor solidity means. Try prising the balloon
    # off what it touches and keep whichever reading scores best: the separated
    # balloon beats the sprawl on form and on how much of it is lettering.
    #
    # Only worth doing when the flood really is straggling. A balloon that came
    # back clean is already the answer, and opening a page-sized mask three
    # times over is not free.
    shape = _region_features(region, _fill_holes(region), page_area)
    if shape is not None and shape["solidity"] < 0.75:
        for candidate in _isolate_blob(region, seed[0], seed[1]):
            other = judge(candidate)
            if other is not None and (best is None or other[0] > best[0]):
                best = other

    if best is not None and best[0] >= min_score:
        return best
    return None


def _piece_candidate(
    piece: np.ndarray, score: float, click: tuple[int, int], min_area_px: int
):
    """Package a split piece as a candidate, carrying a score already earned.

    Only the things that stay true however the region was measured: it has to
    be big enough to letter, and it has to be the piece under the cursor.
    """
    if int(cv2.countNonZero(piece)) < min_area_px:
        return None

    bx, by, bw, bh = cv2.boundingRect(piece)
    cx, cy = click
    if not (bx <= cx < bx + bw and by <= cy < by + bh):
        return None

    filled_crop = _fill_holes(piece[by : by + bh, bx : bx + bw])
    if not filled_crop[cy - by, cx - bx]:
        return None
    return score, (bx, by), filled_crop


def _isolate_blob(region: np.ndarray, seed_x: int, seed_y: int):
    """Separate the compact body under the click from thinner things it touches.

    The opening radius comes from how thick the region is *at the click* -- the
    largest disc that fits there -- not from the bounding box of the whole
    flood. On a balloon fused with half a page of artwork those are wildly
    different numbers, and sizing from the bounding box opens with a radius that
    erases the balloon along with everything else.

    Anything thinner than the balloon falls away; the balloon survives, is taken
    as the component under the click, and is dilated back to its true edge.
    """
    # Measure and open the region with its holes filled. Inside a balloon
    # lettered in white, the fill itself is only a thin web between the letters,
    # so thickness measured on the raw flood reports the gap between two strokes
    # rather than the balloon -- and opens with a radius far too small to
    # separate anything.
    filled = _fill_holes(region)
    distance = cv2.distanceTransform((filled > 0).astype(np.uint8), cv2.DIST_L2, 5)
    reach = float(distance[seed_y, seed_x])
    if reach < 4.0:
        return  # nothing thick enough here to be a balloon body

    # Do the morphology at a reduced scale. Opening a 6.5 MP mask with a 400px
    # kernel takes the best part of a second, and the shapes being separated are
    # hundreds of pixels across -- far coarser than the resolution they are
    # being measured at.
    scale = max(1, int(reach / 24))
    height, width = filled.shape[:2]
    if scale > 1:
        work = cv2.resize(
            filled, (max(1, width // scale), max(1, height // scale)),
            interpolation=cv2.INTER_NEAREST,
        )
        wx = min(work.shape[1] - 1, seed_x // scale)
        wy = min(work.shape[0] - 1, seed_y // scale)
    else:
        work, wx, wy = filled, seed_x, seed_y

    for factor in (0.85, 0.65, 0.45):
        radius = max(2, int((reach / scale) * factor))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
        )
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel)
        if not opened[wy, wx]:
            continue

        count, labels = cv2.connectedComponents((opened > 0).astype(np.uint8))
        if count <= 1:
            continue
        label = labels[wy, wx]
        if label == 0:
            continue

        body = ((labels == label).astype(np.uint8)) * 255
        grown = cv2.dilate(body, kernel)
        if scale > 1:
            grown = cv2.resize(grown, (width, height), interpolation=cv2.INTER_NEAREST)
        yield cv2.bitwise_and(region, cv2.bitwise_and(grown, filled))


def describe_detection(gray: np.ndarray, x: int, y: int) -> str:
    """Explain what a click looked like, for when detection refuses.

    Reports the best candidate actually considered -- including the ones tried
    after prising a balloon off surrounding artwork -- and which requirement it
    missed, rather than only the raw flood.
    """
    h, w = gray.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return "outside the page"

    page_area = h * w
    level = page_threshold(gray)
    started_dark = bool(gray[int(y), int(x)] < level)
    notes = []

    for dark in (started_dark, not started_dark):
        field, sealings = sealed_fill(gray, dark, level)
        sealed, seal = sealings[0], _SEALS[0]
        best = None

        for seed in _seed_candidates(sealed, int(x), int(y), _search_radius(gray)):
            flood = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(
                sealed.copy(), flood, seed, 128, 0, 0,
                4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8),
            )
            extent = _fill_holes(cv2.dilate(flood[1:-1, 1:-1], seal))
            region = cv2.bitwise_and(field, extent)
            for candidate in [region] + list(_isolate_blob(region, seed[0], seed[1])):
                bx, by, bw, bh = cv2.boundingRect(candidate)
                if bw == 0 or bh == 0:
                    continue
                crop = candidate[by : by + bh, bx : bx + bw]
                filled_crop = _fill_holes(crop)
                features = _region_features(crop, filled_crop, page_area)
                if features is None:
                    continue

                covers = (
                    bx <= x < bx + bw
                    and by <= y < by + bh
                    and bool(filled_crop[y - by, x - bx])
                )
                area = int(cv2.countNonZero(candidate))
                reasons = []
                if area > MAX_AREA_FRACTION * page_area:
                    reasons.append("too big")
                if features["holes"] < MIN_HOLES:
                    reasons.append("too few marks")
                if not (MIN_INK <= features["ink"] <= MAX_INK):
                    reasons.append("wrong amount of lettering")
                if not covers:
                    reasons.append("does not cover the click")
                score = _balloon_score(features)
                if not reasons and score < MIN_BALLOON_SCORE:
                    reasons.append("score too low")

                entry = (score, features, reasons)
                if best is None or score > best[0]:
                    best = entry

        if best is None:
            notes.append(f"{'dark' if dark else 'light'}: nothing enclosed")
            continue

        score, features, reasons = best
        detail = ", ".join(
            "%s %.4g" % (name, features[name]) for name in ("ink", *FEATURE_RAMPS)
        )
        notes.append(
            "%s: best %.2f/%.1f (need %.1f)%s [%s]"
            % (
                "dark" if dark else "light",
                score,
                INK_WEIGHT
                + COARSENESS_WEIGHT
                + sum(w for _, _, w in FEATURE_RAMPS.values()),
                MIN_BALLOON_SCORE,
                (" — " + "; ".join(reasons)) if reasons else " — accepted",
                detail,
            )
        )
    return " | ".join(notes)


def layout_region(mask: np.ndarray, click_x: int, click_y: int) -> np.ndarray:
    """Strip the tail off a balloon mask, leaving the body that text should sit in.

    Opening removes the narrow tail, the component under the click is kept, and
    dilating back and re-intersecting restores the body's true edges instead of
    the shrunken ones opening leaves behind.
    """
    _, _, bw, bh = mask_bounds(mask)
    radius = max(6, int(min(bw, bh) * 0.10))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))

    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    count, labels = cv2.connectedComponents((opened > 0).astype(np.uint8))
    if count <= 1:
        return mask

    label = labels[click_y, click_x] if opened[click_y, click_x] else 0
    if label == 0:
        # Click sat in the tail or in trimmed-away edging; take the biggest body.
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        label = int(np.argmax(sizes))

    body = ((labels == label).astype(np.uint8)) * 255
    grown = cv2.dilate(body, kernel)
    return cv2.bitwise_and(grown, mask)


def move_edges(
    mask: np.ndarray,
    left: float = 0.0,
    right: float = 0.0,
    top: float = 0.0,
    bottom: float = 0.0,
) -> np.ndarray:
    """Push each edge of a region outward (positive) or inward (negative).

    Detection gives the balloon the artwork draws, which is not always the area
    that needs covering: original lettering can sit a little outside it, and a
    balloon can be found slightly wide on one side. Moving one edge is the small
    correction that saves erasing a block by hand.

    Done with one-sided structuring elements rather than by clipping to a
    rectangle, so it works on a balloon's real outline -- a spiky shout keeps
    its spikes, and only the side asked for moves.
    """
    edges = (
        (left, True, True),
        (right, True, False),
        (top, False, True),
        (bottom, False, False),
    )
    for amount, horizontal, leading in edges:
        span = int(round(abs(amount)))
        if span < 1:
            continue

        # Which end of the kernel sits on the pixel decides which way the shape
        # grows. For a leading edge (left, top) growing outward looks backwards
        # from the trailing ones, which is why this is a table rather than a sum.
        outward = amount > 0
        index = 0 if (leading == outward) else span
        size = (span + 1, 1) if horizontal else (1, span + 1)
        anchor = (index, 0) if horizontal else (0, index)
        kernel = np.ones((size[1], size[0]), np.uint8)
        operation = cv2.dilate if outward else cv2.erode
        mask = operation(mask, kernel, anchor=anchor)
    return mask


def mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Tight bounding box of a mask as (x, y, w, h)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def row_spans(mask: np.ndarray, row: int, pivot_x: int) -> tuple[int, int] | None:
    """The horizontal run of set pixels on `row` that contains or is nearest `pivot_x`.

    Used by the layout engine to measure how much width a line of text has
    available at a given height inside a non-rectangular balloon.
    """
    if not (0 <= row < mask.shape[0]):
        return None
    line = mask[row]
    if not line.any():
        return None

    # Boundaries of every run of set pixels on this scanline.
    padded = np.concatenate(([0], (line > 0).view(np.uint8), [0]))
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)  # exclusive

    best = None
    best_dist = None
    for s, e in zip(starts, ends):
        if s <= pivot_x < e:
            return (int(s), int(e))
        dist = s - pivot_x if pivot_x < s else pivot_x - (e - 1)
        if best_dist is None or dist < best_dist:
            best, best_dist = (int(s), int(e)), dist
    return best
