"""Drawing text boxes onto a painter, for both the editor and full-resolution export."""

from __future__ import annotations

import math

import cv2
import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QImage,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPixmap,
    QRawFont,
    QTransform,
)

from .imaging import detect_bubble, layout_region, mask_bounds, move_edges
from .model import DEFAULT_FONT_CANDIDATES, TextBox
from .repair import repair_region
from .textlayout import (
    LayoutTarget,
    fit_font_size,
    fit_font_size_vertical,
    layout,
    layout_vertical,
)


# The family new boxes start in, and the one to fall back to when a box asks
# for a family this machine does not have. Set from the saved preference at
# startup; empty means "no choice made, use the built-in order".
_chosen_font = ""


def set_chosen_font(family: str) -> None:
    """Choose the lettering font: for new boxes, and as the fallback."""
    global _chosen_font
    _chosen_font = family or ""
    _coverage_cache.clear()  # the CJK fallback was resolved against the old one


def default_font() -> str:
    """A stand-in to letter in until one is chosen."""
    families = set(QFontDatabase.families())
    for candidate in DEFAULT_FONT_CANDIDATES:
        if candidate in families:
            return candidate
    return QFontDatabase.systemFont(QFontDatabase.GeneralFont).family()


def chosen_font() -> str:
    """The deliberately chosen lettering font, or "" if none has been.

    New boxes record this rather than the stand-in. Writing the stand-in into a
    box would freeze it there: choose a real font later and every box made
    before the choice would still be asking for whatever happened to be
    installed on the day.
    """
    return _chosen_font


def available_font(family: str) -> str:
    """Resolve a font family, falling back when it is not installed."""
    families = set(QFontDatabase.families())
    if family in families:
        return family
    # A deliberate choice outranks the built-in preference: a page lettered
    # before the font was picked asks for a family that is not here, and it
    # should come back in the font being used now rather than in a stand-in.
    if _chosen_font and _chosen_font in families:
        return _chosen_font
    return default_font()


# Tried in order when the chosen font has no glyphs for the text.
CJK_FALLBACKS = (
    "Noto Sans JP",
    "Yu Gothic",
    "Meiryo",
    "MS Gothic",
    "BIZ UDGothic",
    "Noto Sans KR",
    "Malgun Gothic",
    "SimSun",
)

HANGUL_FALLBACKS = ("Noto Sans KR", "Malgun Gothic", "Batang")

_coverage_cache: dict = {}


def font_for_text(family: str, text: str) -> str:
    """Swap in a font that can actually draw `text`.

    Text is rendered through QPainterPath, which draws only what the chosen font
    provides -- there is no automatic per-character fallback as there would be in
    a label. A Latin comic font asked for Japanese therefore produces a row of
    identical missing-glyph boxes rather than words.
    """
    probe = next((ch for ch in text if not ch.isspace() and ord(ch) > 0x2E80), None)
    if probe is None:
        return family

    key = (family, probe)
    cached = _coverage_cache.get(key)
    if cached is not None:
        return cached

    result = family
    if not QRawFont.fromFont(QFont(family)).supportsCharacter(probe):
        families = set(QFontDatabase.families())
        # Hangul first for Korean: several Japanese fonts carry a partial Hangul
        # set, and picking one gives passable-looking but wrong-looking text.
        order = (
            HANGUL_FALLBACKS + CJK_FALLBACKS
            if 0xAC00 <= ord(probe) <= 0xD7AF
            else CJK_FALLBACKS
        )
        for candidate in order:
            if candidate in families and QRawFont.fromFont(
                QFont(candidate)
            ).supportsCharacter(probe):
                result = candidate
                break

    _coverage_cache[key] = result
    return result


def build_font(box: TextBox, size: float | None = None) -> QFont:
    font = QFont(font_for_text(available_font(box.font_family), box.text))
    font.setPointSizeF(size if size is not None else box.font_size)
    font.setBold(box.bold)
    font.setItalic(box.italic)
    font.setHintingPreference(QFont.PreferNoHinting)
    return font


def mask_to_path(mask: np.ndarray) -> QPainterPath:
    """Trace a mask into a fillable outline.

    Drawing the erased region as an image meant a scaled QImage blit every
    frame, and Qt's smooth downscale path for QImage costs ~26 ms per balloon --
    on its own enough to make panning stutter. A traced path fills in well under
    a millisecond and stays crisp at any zoom.
    """
    path = QPainterPath()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        points = contour.reshape(-1, 2)
        if len(points) < 3:
            continue
        path.moveTo(float(points[0][0]), float(points[0][1]))
        for px, py in points[1:]:
            path.lineTo(float(px), float(py))
        path.closeSubpath()
    return path


def build_text_path(lines, font: QFont) -> QPainterPath:
    """Trace laid-out lines into a single glyph outline."""
    path = QPainterPath()
    for line in lines:
        if line.glyphs:
            # A vertical column places every character itself, and turns a few
            # of them a quarter turn -- brackets and the long-vowel mark point
            # along the column rather than across it.
            for glyph in line.glyphs:
                if not glyph.text.strip():
                    continue
                if glyph.rotated:
                    turned = QPainterPath()
                    turned.addText(0.0, 0.0, font, glyph.text)
                    centre = turned.boundingRect().center()
                    transform = (
                        QTransform()
                        .translate(glyph.x + centre.x(), glyph.baseline + centre.y())
                        .rotate(90)
                        .translate(-centre.x(), -centre.y())
                    )
                    path.addPath(transform.map(turned))
                else:
                    path.addText(glyph.x, glyph.baseline, font, glyph.text)
        elif line.text:
            path.addText(line.x, line.baseline, font, line.text)
    return path


def outline_band(path: QPainterPath, outline_width: float) -> QPainterPath:
    """A ribbon straddling the glyph edge, to be drawn under the glyphs.

    The outline has to grow strictly outward: a centred pen stroke drawn *over*
    the text eats into the letterform as it widens, which chokes thin comic
    lettering. Filling this band first and the glyphs on top achieves that, and
    leaves no seam -- the glyph's antialiased edge lands in the middle of solid
    outline colour, not against it.

    Uniting the band with the glyph path gives a pixel-equivalent result and was
    what this did originally, but QPainterPath's boolean union is quadratic-ish
    in outline complexity: a balloon's worth of text cost ~350 ms to unite and
    only ~2 ms to fill, and that ran on every repaint of every box. Winding fill
    gets the same picture from overlapping subpaths for free.
    """
    stroker = QPainterPathStroker()
    stroker.setWidth(outline_width * 2.0)
    stroker.setJoinStyle(Qt.RoundJoin)
    stroker.setCapStyle(Qt.RoundCap)
    band = stroker.createStroke(path)
    # Adjacent letters' bands overlap, and odd-even fill would punch those
    # overlaps back out into holes.
    band.setFillRule(Qt.WindingFill)
    return band


class BoxRuntime:
    """Cached geometry and layout for one text box.

    Masks and per-row spans are expensive to rebuild at 600 dpi, so they are
    derived once and invalidated only when the inputs that affect them change.
    """

    def __init__(self, box: TextBox, gray: np.ndarray, colour: np.ndarray | None = None):
        self.box = box
        self.gray = gray
        # Shape and density come off the grey page; the pixels that end up on
        # the page come off the colour one. Callers without a colour page get a
        # grey one promoted, so a black-and-white comic behaves as it always did.
        self.colour = (
            colour if colour is not None else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        )
        self.mask: np.ndarray | None = None
        # What gets covered, which is the balloon with its edges nudged. Kept
        # apart from `mask`, which stays the shape the artwork drew and is what
        # hit testing, colour sampling and layout all still work from.
        self.erase_mask: np.ndarray | None = None
        self.erase_path: QPainterPath | None = None
        self.bubble_dark = False  # balloon filled black, lettered in white
        self.fill_colour = QColor("white")  # what a flat erase should paint
        self.repair = None  # Repair | None -- reconstructed background
        self.repair_pixmap: QPixmap | None = None
        self.repair_note = ""
        self._repair_dirty = True
        self.target: LayoutTarget | None = None
        self.transposed: LayoutTarget | None = None
        self.lines: list = []
        self.effective_size: float = box.font_size
        self.overflow = False
        self._layout_key = None
        # Traced glyph outline and its outline band, kept between repaints. The
        # page redraws on every caret blink and every scroll step, and retracing
        # six balloons' worth of text each time is the difference between a
        # responsive page and a page that stutters once it has text on it.
        self._paint_key = None
        self._text_path: QPainterPath | None = None
        self._band: QPainterPath | None = None
        self._target_key = None  # invalidated whenever the mask changes
        self._bounds: tuple[int, int, int, int] | None = None
        self.rebuild()

    # -- geometry -------------------------------------------------------

    def rebuild(self) -> None:
        """Recompute the mask and everything downstream of it."""
        self.bubble_dark = False
        self.mask = self._build_mask()
        self.fill_colour = self._flat_colour()
        self._bounds = None
        self.refresh_erase()
        # Reconstruction costs a couple of hundred milliseconds, so it is not
        # done here -- dragging a box would recompute it on every mouse move.
        self.repair = None
        self.repair_pixmap = None
        self._repair_dirty = True
        self._layout_key = None
        self._target_key = None  # the region itself just changed shape
        self.relayout()

    def refresh_erase(self) -> None:
        """Recompute the covered area after its edges move.

        Separate from rebuild() because nudging an edge must not re-run balloon
        detection -- that costs a couple of hundred milliseconds, which is the
        difference between a control you can feel your way with and one you
        wait on.
        """
        box = self.box
        if self.mask is None:
            self.erase_mask = None
            self.erase_path = None
            return

        self.erase_mask = move_edges(
            self.mask,
            box.erase_left,
            box.erase_right,
            box.erase_top,
            box.erase_bottom,
        )
        self.erase_path = mask_to_path(self.erase_mask)
        self._repair_dirty = True

    def ensure_repair(self) -> None:
        """Reconstruct the background under this box, if it needs one.

        Call once an interaction has settled, not while dragging.
        """
        if not self._repair_dirty:
            return
        self._repair_dirty = False
        self.repair = None
        self.repair_pixmap = None
        self.repair_note = ""

        box = self.box
        if not box.erase or box.erase_mode != "rebuild" or self.erase_mask is None:
            return

        result = repair_region(self.gray, self.erase_mask, self.colour)
        if result is None:
            self.repair_note = "background too detailed to rebuild"
            return

        patch = np.ascontiguousarray(result.patch)
        height, width = patch.shape[:2]
        # BGR888 rather than RGB888: the page is already in OpenCV's order, and
        # swapping it here would turn every rebuilt patch's reds into blues.
        image = QImage(patch.data, width, height, 3 * width, QImage.Format_BGR888)
        self.repair = result
        self.repair_pixmap = QPixmap.fromImage(image.copy())
        self.repair_note = result.note

    def _flat_colour(self) -> QColor:
        """What a flat erase should paint: the colour the region is made of.

        Taken as the median over the region, so the lettering inside it -- a
        minority of the pixels, and the thing being covered up -- does not drag
        it. White paper gives white, an inverted balloon gives black, and a
        caption over a coloured panel gives that colour instead of a hole.
        """
        if self.mask is None:
            return QColor("black" if self.bubble_dark else "white")

        x, y, w, h = mask_bounds(self.mask)
        if not (w and h):
            return QColor("black" if self.bubble_dark else "white")

        crop = self.mask[y : y + h, x : x + w]
        pixels = self.colour[y : y + h, x : x + w][crop > 0]
        if pixels.size == 0:
            return QColor("black" if self.bubble_dark else "white")

        blue, green, red = np.median(pixels.reshape(-1, 3), axis=0).astype(int)
        return QColor(int(red), int(green), int(blue))

    def _build_mask(self) -> np.ndarray | None:
        box = self.box
        if box.kind == "bubble" and box.seed:
            x, y = int(box.seed[0]), int(box.seed[1])
            found = detect_bubble(self.gray, x, y)
            if found is None:
                return None
            mask, dark = found
            self.bubble_dark = dark
            return layout_region(mask, x, y)

        if box.rect:
            x, y, w, h = (int(round(v)) for v in box.rect)
            mask = np.zeros(self.gray.shape[:2], dtype=np.uint8)
            H, W = mask.shape
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(W, x + w), min(H, y + h)
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = 255
                # A flat fill has to match the paper it covers. Hand-drawn boxes
                # exist mostly for balloons detection could not find, and the
                # ones it cannot find are disproportionately the black ones --
                # filling those with white would punch a hole straight through
                # the balloon.
                under = self.gray[y0:y1, x0:x1]
                self.bubble_dark = bool(np.median(under) < 128)
                return mask
        return None

    @property
    def centre(self) -> tuple[float, float]:
        x, y, w, h = self.bounds
        return (x + w / 2.0, y + h / 2.0)

    def apply_transform(self, painter: QPainter) -> None:
        """Move the painter into the box's own frame: nudged, then rotated."""
        box = self.box
        painter.translate(box.offset[0], box.offset[1])
        if box.angle:
            cx, cy = self.centre
            painter.translate(cx, cy)
            painter.rotate(box.angle)
            painter.translate(-cx, -cy)

    def to_local(self, px: float, py: float) -> tuple[float, float]:
        """Map a page point back into the box's own frame.

        Hit testing and caret placement work against the unrotated layout, so
        clicks have to be carried back through the same transform the text was
        drawn with.
        """
        box = self.box
        cx, cy = self.centre
        qx = px - box.offset[0] - cx
        qy = py - box.offset[1] - cy
        if box.angle:
            radians = math.radians(-box.angle)
            cos, sin = math.cos(radians), math.sin(radians)
            qx, qy = qx * cos - qy * sin, qx * sin + qy * cos
        return (cx + qx, cy + qy)

    def text_area_rect(self) -> QRectF:
        """The area text is allowed to flow into, before the box's transform.

        What the side handles stretch. It is the balloon inset by the padding
        and adjusted by the stretch, so it moves as those change -- unlike the
        region itself, which is fixed by the artwork.
        """
        if self.target is not None and self.target.valid:
            return QRectF(
                self.target.x0, self.target.y0, self.target.width, self.target.height
            )
        x, y, w, h = self.bounds
        return QRectF(x, y, w, h)

    def frame_rect(self) -> QRectF:
        """The rectangle handles hang off, before the box's own transform.

        The lettering rather than the region: a corner drag scales the text, so
        the frame should hug the text. Falling back to the region keeps a box
        with nothing in it grabbable.
        """
        x, y, w, h = self.bounds
        path, _ = self._paths()
        if path is None or path.isEmpty():
            return QRectF(x, y, w, h)
        rect = path.boundingRect()
        # A little air, so the frame does not sit right on the glyph edges.
        margin = max(4.0, self.effective_size * 0.18)
        return rect.adjusted(-margin, -margin, margin, margin)

    def painted_rect(self) -> QRectF:
        """Everything this box puts on the page, in page coordinates.

        Wider than the region when the text is nudged, rotated or overflowing,
        so it can be used to repaint just this box without leaving a trail.
        """
        x, y, w, h = self.bounds
        rect = QRectF(x, y, w, h)

        # The erase can now reach outside the balloon, and anything it covers
        # has to be repainted or a nudged edge leaves a trail behind it.
        if self.erase_path is not None and not self.erase_path.isEmpty():
            rect = rect.united(self.erase_path.boundingRect())

        path, _ = self._paths()
        if path is not None:
            text = path.boundingRect()
            box = self.box
            if box.angle:
                cx, cy = self.centre
                transform = (
                    QTransform().translate(cx, cy).rotate(box.angle).translate(-cx, -cy)
                )
                text = transform.mapRect(text)
            text.translate(box.offset[0], box.offset[1])
            rect = rect.united(text)

        # Room for the outline band, the caret and antialiasing.
        margin = 8.0 + self.box.outline_width * 2.0
        return rect.adjusted(-margin, -margin, margin, margin)

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        # Measured once per mask: mask_bounds scans the whole page, and this is
        # asked for on every repaint, every caret blink and every hit test.
        if self._bounds is None:
            self._bounds = (
                (0, 0, 0, 0) if self.mask is None else mask_bounds(self.mask)
            )
        return self._bounds

    # -- layout ---------------------------------------------------------

    def relayout(self, force: bool = False) -> None:
        box = self.box
        key = (
            box.text,
            box.font_family,
            box.font_size,
            box.auto_fit,
            box.bold,
            box.italic,
            box.line_spacing,
            box.align,
            box.padding,
            box.vertical,
            box.max_lines,
            box.stretch_x,
            box.stretch_y,
        )
        if not force and key == self._layout_key:
            return
        self._layout_key = key
        self._paint_key = None  # the lines are about to change under it

        if self.mask is None:
            self.target, self.lines, self.overflow = None, [], False
            self._target_key = None
            return

        # Measuring the region's rows costs ~35 ms, and it is the same answer
        # for every keystroke: the shape you are typing into depends on the mask
        # and the padding, not on the words. Rebuilding it per edit was most of
        # what made a page with text on it feel heavy.
        target_key = (box.padding, box.vertical, box.stretch_x, box.stretch_y)
        if target_key != self._target_key:
            self._target_key = target_key
            pad_x = box.padding - box.stretch_x
            pad_y = box.padding - box.stretch_y
            self.target = LayoutTarget(self.mask, pad_x, pad_y)
            # Vertical setting asks "how far down can this column run", which is
            # the same question the horizontal engine answers about rows -- on a
            # region with its axes swapped.
            # Transposing swaps the axes, so the insets swap with them.
            self.transposed = (
                LayoutTarget(np.ascontiguousarray(self.mask.T), pad_y, pad_x)
                if box.vertical and self.target.valid
                else None
            )

        if not self.target.valid:
            self.lines, self.overflow = [], False
            return

        vertical = box.vertical and self.transposed is not None

        size = box.font_size
        if box.auto_fit and box.text.strip():
            if vertical:
                size = fit_font_size_vertical(
                    box.text, self.target, self.transposed, build_font(box, 100.0),
                    box.line_spacing, max_lines=box.max_lines,
                )
            else:
                size = fit_font_size(
                    box.text, self.target, build_font(box, 100.0), box.line_spacing,
                    box.align, max_lines=box.max_lines,
                )
        self.effective_size = size

        font = build_font(box, size)
        if vertical:
            laid = layout_vertical(
                box.text, self.target, self.transposed, font, box.line_spacing
            )
        else:
            laid = layout(box.text, self.target, font, box.line_spacing, box.align)

        self.overflow = laid is None
        if laid is None:
            # Too big for the balloon. Show it anyway, spilling over, and let the
            # inspector say so -- text quietly vanishing is never the right answer.
            if vertical:
                laid = layout_vertical(
                    box.text, self.target, self.transposed, font, box.line_spacing,
                    strict=False,
                )
            else:
                laid = layout(
                    box.text, self.target, font, box.line_spacing, box.align, strict=False
                )
        self.lines = laid or []

    # -- painting -------------------------------------------------------

    def paint_erase(self, painter: QPainter) -> None:
        """Clear the original lettering, by fill or by reconstruction."""
        box = self.box
        # Selecting a balloon leaves the page exactly as it was; the original
        # lettering only disappears once an edit has actually been made.
        if not (box.erase and (box.touched or bool(box.text.strip()))):
            return

        if box.erase_mode == "rebuild":
            # Only paint a reconstruction we actually have. If the engine could
            # not read the background, leave the artwork alone -- a white hole
            # is worse than the original lettering showing.
            if self.repair_pixmap is not None:
                ox, oy = self.repair.origin
                painter.drawPixmap(
                    QRectF(ox, oy, self.repair_pixmap.width(), self.repair_pixmap.height()),
                    self.repair_pixmap,
                    QRectF(self.repair_pixmap.rect()),
                )
        elif self.erase_path is not None and not self.erase_path.isEmpty():
            # Clear back to the region's own fill: white paper in an ordinary
            # balloon, black in an inverted one, and whatever colour the paper
            # happens to be when it is not paper at all.
            painter.fillPath(self.erase_path, self.fill_colour)

    def _paths(self) -> tuple[QPainterPath | None, QPainterPath | None]:
        """Glyph outline and outline band, traced once per layout."""
        box = self.box
        key = (self._layout_key, self.effective_size, box.outline_width)
        if key != self._paint_key:
            self._paint_key = key
            path = build_text_path(self.lines, build_font(box, self.effective_size))
            if path.isEmpty():
                self._text_path, self._band = None, None
            else:
                self._text_path = path
                self._band = (
                    outline_band(path, box.outline_width)
                    if box.outline_width > 0
                    else None
                )
        return self._text_path, self._band

    def paint_text(self, painter: QPainter) -> None:
        if not self.lines:
            return

        path, band = self._paths()
        if path is None:
            return

        box = self.box
        painter.save()
        self.apply_transform(painter)
        painter.setRenderHint(QPainter.Antialiasing, True)
        if band is not None:
            painter.fillPath(band, QColor(box.outline_color))
        painter.fillPath(path, QColor(box.color))
        painter.restore()

    def paint(self, painter: QPainter) -> None:
        self.paint_erase(painter)
        self.paint_text(painter)
