"""The editor window: page canvas, text-box interaction, inspector, export."""

from __future__ import annotations

import math
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import (
    QEvent,
    QPoint,
    QPointF,
    QRect,
    QRectF,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QFontDatabase,
    QFontMetricsF,
    QImage,
    QImageReader,
    QKeySequence,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
    QTextBlockFormat,
    QTextCursor,
    QTextOption,
    QTransform,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from .history import History, RepairEntry
from .imaging import (
    describe_detection,
    imread_unicode,
    imwrite_unicode,
    load_colour,
)
from .model import (
    EXPORT_SUFFIX,
    IMAGE_FILTER,
    Project,
    TextBox,
)
from . import ocr, prefs
from .repair import lattice_at
from .repairlayer import RepairLayer
from .render import (
    BoxRuntime,
    available_font,
    build_font,
    chosen_font,
    default_font,
    set_chosen_font,
)
from .textlayout import (
    caret_geometry_vertical,
    caret_x,
    index_at_point,
    index_at_point_vertical,
    line_for_index,
)

KIND_ROLE = Qt.UserRole + 1  # "page" or "folder" on a side-panel row

HANDLE_TINT = QColor(64, 132, 255, 60)
SELECTION_TINT = QColor(64, 132, 255, 90)
CARET_COLOUR = QColor(32, 96, 220)
HANDLE_EDGE = QColor(32, 96, 220)
HANDLE_FILL = QColor(255, 255, 255)
AREA_EDGE = QColor(64, 132, 255, 130)

# Handles are sized in screen pixels and divided by the zoom, so they stay the
# same size to grab whether you are looking at the whole page or one balloon.
HANDLE_SCREEN_SIZE = 9.0
HANDLE_GRAB_SIZE = 14.0  # generous hit area; the drawn square is smaller
ROTATE_SCREEN_GAP = 26.0  # how far the rotation handle floats above the frame

CORNERS = ("tl", "tr", "br", "bl")
EDGES = ("left", "right", "top", "bottom")


def _letter_for_background(runtime: BoxRuntime) -> None:
    """Start a box in a colour that will be visible where it sits.

    Black on a black balloon is invisible against its own fill, and the fill is
    the first thing that happens when you type.
    """
    if runtime.bubble_dark:
        runtime.box.color = "#FFFFFF"
        runtime.box.outline_color = "#000000"


class PageItem(QGraphicsItem):
    """The page image, drawn from a pyramid of pre-scaled copies.

    Smoothly rescaling a 2150x3035 pixmap on every repaint is what makes
    scrolling crawl, but drawing it without smoothing turns the screentone into
    moire. Halving the page a few times up front gives both: pick the level
    closest to the on-screen size and the per-frame resample is then tiny.
    """

    MAX_LEVELS = 6

    def __init__(self, pixmap: QPixmap):
        super().__init__()
        self.levels = [pixmap]
        while (
            len(self.levels) < self.MAX_LEVELS
            and self.levels[-1].width() > 256
            and self.levels[-1].height() > 256
        ):
            previous = self.levels[-1]
            self.levels.append(
                previous.scaled(
                    previous.width() // 2,
                    previous.height() // 2,
                    Qt.IgnoreAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

    def boundingRect(self) -> QRectF:
        return QRectF(self.levels[0].rect())

    def paint(self, painter: QPainter, option, widget=None) -> None:
        scale = abs(painter.transform().m11()) or 1.0

        level = 0
        while level + 1 < len(self.levels) and scale <= 0.5 ** (level + 1) * 2.0:
            level += 1

        pixmap = self.levels[level]
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawPixmap(self.boundingRect(), pixmap, QRectF(pixmap.rect()))


class LetteringItem(QGraphicsItem):
    """Paints every text box on the current page.

    One item for the whole page rather than an item per box: hit testing is done
    against the real masks, which is both exact for odd balloon shapes and far
    less bookkeeping than keeping per-item bounding rects in sync.
    """

    def __init__(self, page_size):
        super().__init__()
        self._size = page_size
        self.runtimes: list[BoxRuntime] = []
        self.selected_id: str | None = None
        # (runtime, selection_start, selection_end, caret_index, caret_visible)
        self.edit: tuple | None = None
        self.repair: RepairLayer | None = None
        self.repair_pixmap: QPixmap | None = None
        self.hidden = False  # show the page as it arrived, lettering and all off

    def refresh_repair(self) -> None:
        """Re-cache the painted region for display.

        Only the touched area is converted, not the whole 6.5 MP layer.
        """
        if self.repair is None or self.repair.dirty is None:
            self.repair_pixmap = None
            return
        x0, y0, x1, y1 = self.repair.dirty
        self.repair_pixmap = QPixmap.fromImage(self.repair.region_image(x0, y0, x1, y1))

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._size[0], self._size[1])

    def paint(self, painter: QPainter, option, widget=None) -> None:
        # Everything this item draws, gone: the erases and repairs as well as the
        # text, because the point of hiding is to read the original underneath
        # and a white patch over it would defeat that. The page item below is
        # untouched artwork, so this shows the page exactly as it arrived.
        if self.hidden:
            return

        # Three passes, because manual repairs belong between the automatic
        # ones and the lettering: you use the stamp to fix what the rebuild got
        # wrong, and neither should be painted over by the other.
        for runtime in self.runtimes:
            runtime.paint_erase(painter)

        if self.repair is not None and self.repair_pixmap is not None:
            x0, y0, x1, y1 = self.repair.dirty
            painter.drawPixmap(
                QRectF(x0, y0, x1 - x0, y1 - y0),
                self.repair_pixmap,
                QRectF(self.repair_pixmap.rect()),
            )

        for runtime in self.runtimes:
            runtime.paint_text(painter)

        self._paint_caret(painter)

        selected = self.runtime_for(self.selected_id)
        if selected is not None and selected.mask is not None:
            x, y, w, h = selected.bounds
            if w and h:
                # No tint over text being edited -- the caret already says where
                # you are, and a wash of blue over the words defeats the point of
                # showing the real thing.
                editing = self.edit is not None and self.edit[0] is selected
                # Solid rather than dashed; cosmetic dashed pens are a known
                # slow path in the raster engine.
                painter.setBrush(Qt.NoBrush if editing else HANDLE_TINT)
                painter.setPen(QPen(QColor(64, 132, 255), 0, Qt.SolidLine))
                painter.drawRect(QRectF(x, y, w, h))
                if not editing:
                    self._paint_handles(painter, selected)

    def _paint_handles(self, painter: QPainter, runtime: BoxRuntime) -> None:
        """Corner handles to scale the text, and one above it to rotate."""
        scale = abs(painter.transform().m11()) or 1.0
        points = self.handle_points(runtime, scale)
        if not points:
            return

        size = HANDLE_SCREEN_SIZE / scale
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(HANDLE_EDGE, 0, Qt.SolidLine))

        # The frame itself, following the text rather than sitting upright
        # around it, so a rotated box reads as rotated.
        painter.setBrush(Qt.NoBrush)
        painter.drawPolygon([points[name] for name in CORNERS])
        painter.drawLine(points["anchor"], points["rotate"])

        # The text area, drawn faintly behind its own handles so it is clear
        # what the sides move.
        painter.setPen(QPen(AREA_EDGE, 0, Qt.SolidLine))
        painter.drawPolygon(
            [points[name] for name in ("area_tl", "area_tr", "area_br", "area_bl")]
        )

        painter.setBrush(HANDLE_FILL)
        painter.setPen(QPen(AREA_EDGE, 0, Qt.SolidLine))
        for name in EDGES:
            point = points[name]
            painter.drawEllipse(point, size * 0.45, size * 0.45)

        painter.setPen(QPen(HANDLE_EDGE, 0, Qt.SolidLine))
        for name in CORNERS:
            point = points[name]
            painter.drawRect(
                QRectF(point.x() - size / 2, point.y() - size / 2, size, size)
            )
        centre = points["rotate"]
        painter.drawEllipse(centre, size * 0.6, size * 0.6)
        painter.restore()

    def _paint_caret(self, painter: QPainter) -> None:
        """Draw the caret and selection onto the laid-out text itself."""
        if self.edit is None:
            return
        runtime, start, end, caret, caret_visible = self.edit
        metrics = QFontMetricsF(build_font(runtime.box, runtime.effective_size))
        ascent, height = metrics.ascent(), metrics.height()

        painter.save()
        runtime.apply_transform(painter)

        if not runtime.lines:
            # Nothing typed yet: park the caret in the middle of the region so
            # there is somewhere visible to start from.
            if caret_visible and runtime.target is not None and runtime.target.valid:
                width = max(1.5, height * 0.07)
                painter.fillRect(
                    QRectF(
                        runtime.target.centre_x - width / 2,
                        runtime.target.centre_y - height / 2,
                        width,
                        height,
                    ),
                    CARET_COLOUR,
                )
            painter.restore()
            return

        if runtime.box.vertical:
            column = metrics.height() * runtime.box.line_spacing
            if start != end:
                for line in runtime.lines:
                    for glyph in line.glyphs:
                        if start <= glyph.start < end:
                            painter.fillRect(
                                QRectF(line.x, glyph.baseline - ascent, column, height),
                                SELECTION_TINT,
                            )
            if caret_visible:
                line = runtime.lines[line_for_index(runtime.lines, caret)]
                painter.fillRect(
                    QRectF(*caret_geometry_vertical(line, metrics, caret, column)),
                    CARET_COLOUR,
                )
            painter.restore()
            return

        if start != end:
            for line in runtime.lines:
                if end <= line.start or start >= line.end:
                    continue
                left = caret_x(line, metrics, max(start, line.start))
                right = caret_x(line, metrics, min(end, line.end))
                painter.fillRect(
                    QRectF(left, line.baseline - ascent, max(right - left, 2.0), height),
                    SELECTION_TINT,
                )

        if caret_visible:
            line = runtime.lines[line_for_index(runtime.lines, caret)]
            x = caret_x(line, metrics, caret)
            width = max(1.5, height * 0.07)
            painter.fillRect(
                QRectF(x - width / 2, line.baseline - ascent, width, height), CARET_COLOUR
            )

        painter.restore()

    def handle_points(self, runtime: BoxRuntime, scale: float) -> dict:
        """Where the drag handles sit, in page coordinates.

        One definition, used both to draw them and to decide what was grabbed --
        the two drifting apart is how handles end up not where they look.

        The frame hugs the lettering, not the balloon: a corner drag scales the
        text, so that is what it should appear to grip. It is carried through
        the same nudge and rotation the text is drawn with, which also keeps it
        from competing with the region tint once the box is turned.
        """
        frame = runtime.frame_rect()
        x, y, w, h = frame.x(), frame.y(), frame.width(), frame.height()
        if not (w and h):
            return {}

        box = runtime.box
        transform = QTransform().translate(box.offset[0], box.offset[1])
        if box.angle:
            cx, cy = runtime.centre
            transform.translate(cx, cy)
            transform.rotate(box.angle)
            transform.translate(-cx, -cy)

        points = {
            "tl": QPointF(x, y),
            "tr": QPointF(x + w, y),
            "br": QPointF(x + w, y + h),
            "bl": QPointF(x, y + h),
        }
        # Above the top edge in the frame's own direction, so it stays "up" as
        # the text turns.
        gap = ROTATE_SCREEN_GAP / max(scale, 1e-6)
        points["rotate"] = QPointF(x + w / 2, y - gap)
        points["anchor"] = QPointF(x + w / 2, y)

        # The sides belong to the text area, not to the text: they stretch the
        # room the words have, and that room is what you are judging when a
        # word falls to a line of its own.
        area = runtime.text_area_rect()
        points["left"] = QPointF(area.left(), area.center().y())
        points["right"] = QPointF(area.right(), area.center().y())
        points["top"] = QPointF(area.center().x(), area.top())
        points["bottom"] = QPointF(area.center().x(), area.bottom())
        points["area_tl"] = QPointF(area.left(), area.top())
        points["area_tr"] = QPointF(area.right(), area.top())
        points["area_br"] = QPointF(area.right(), area.bottom())
        points["area_bl"] = QPointF(area.left(), area.bottom())
        return {name: transform.map(point) for name, point in points.items()}

    def runtime_for(self, box_id: str | None) -> BoxRuntime | None:
        if box_id is None:
            return None
        return next((r for r in self.runtimes if r.box.id == box_id), None)

    def hit_test(self, x: float, y: float) -> BoxRuntime | None:
        """Topmost box covering this point.

        Tests the region both where it sits and where its text has been nudged
        to, so text moved off its balloon can still be grabbed where you see it.
        """
        for runtime in reversed(self.runtimes):
            mask = runtime.mask
            if mask is None:
                continue
            for px, py in ((x, y), runtime.to_local(x, y)):
                xi, yi = int(round(px)), int(round(py))
                if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and mask[yi, xi]:
                    return runtime
        return None


class PageView(QGraphicsView):
    """Zoomable page canvas with click-to-select and double-click-to-edit."""

    selection_changed = Signal(object)  # BoxRuntime | None
    box_requested = Signal(int, int)  # click that did not land on an existing box
    box_drawn = Signal(int, int, int, int)  # a box drawn by hand
    text_edited = Signal()
    snapshot_requested = Signal(str, object)  # label, coalesce key -- before changing
    editing_finished = Signal()  # ends the current undo run
    stroke_finished = Signal(object)  # tile backup from one clone-stamp stroke
    snapped_to = Signal(int, int)  # lattice the clone offset was aligned to
    brush_resized = Signal(int)
    handle_dragged = Signal(object)  # BoxRuntime being scaled or rotated
    lettering_hidden_changed = Signal(bool)
    text_wanted = Signal(object)  # BoxRuntime, or (x, y, w, h) to read
    menu_wanted = Signal(object, object)  # screen point, BoxRuntime or None

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QColor(52, 54, 58))
        self.setFocusPolicy(Qt.StrongFocus)

        self.lettering: LetteringItem | None = None
        self._pan_from: QPoint | None = None
        self._drag_from = None
        self._drag_runtime: BoxRuntime | None = None
        self._drag_origin = None
        self._handle_drag: str | None = None
        self._handle_runtime: BoxRuntime | None = None
        self._handle_centre = None
        self._handle_origin = None
        self._stretch_origin = None
        self._hover_handle: str | None = None
        self.lettering_hidden = False
        self.lookup_active = False   # drag a rectangle to read its text

        self.editor = QTextEdit(self.viewport())
        self.editor.hide()
        self.editor.setFrameStyle(0)
        self.editor.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.editor.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.editor.setAcceptRichText(False)
        # Invisible: it exists only to own the text cursor, key handling, undo
        # and IME. What you see and edit is the real shape-fitted render on the
        # canvas, with the caret drawn there.
        self.editor.setStyleSheet(
            "QTextEdit { background: transparent; border: none; color: transparent; }"
        )
        self.editor.setCursorWidth(0)
        self.editor.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        # A stylesheet alone is not enough: the native selection highlight is
        # painted from the palette and would show through.
        palette = self.editor.palette()
        for role in (
            QPalette.ColorRole.Highlight,
            QPalette.ColorRole.HighlightedText,
            QPalette.ColorRole.Text,
            QPalette.ColorRole.Base,
        ):
            palette.setColor(role, Qt.transparent)
        self.editor.setPalette(palette)
        self.editor.textChanged.connect(self._on_editor_changed)
        self.editor.cursorPositionChanged.connect(self._sync_edit_state)
        self.editor.installEventFilter(self)
        self.editor.selectionChanged.connect(self._sync_edit_state)
        self._editing: BoxRuntime | None = None
        self._last_double_click = (0.0, None)  # (timestamp, viewport point)
        self._drawing_from = None
        self._drawing_to = None

        # Clone stamp
        self.stamp_active = False
        self.stamp_radius = 24
        self.stamp_snap = True
        self.stamp_source: tuple[int, int] | None = None
        self.stamp_offset: tuple[int, int] | None = None
        self._stamp_backup: dict | None = None
        self._stamp_last: tuple[int, int] | None = None
        self._cursor_point = None
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self._editor_base_rect: QRect | None = None
        self._selecting = False

        self._caret_visible = True
        self._caret_timer = QTimer(self)
        self._caret_timer.setInterval(530)
        self._caret_timer.timeout.connect(self._blink_caret)

    # -- page loading ---------------------------------------------------

    def show_page(self, pixmap: QPixmap, lettering: LetteringItem) -> None:
        self.commit_edit()
        scene = self.scene()
        scene.clear()
        scene.setSceneRect(QRectF(pixmap.rect()))
        page_item = PageItem(pixmap)
        page_item.setZValue(0)
        scene.addItem(page_item)
        lettering.setZValue(1)
        scene.addItem(lettering)
        # Hiding is a way of looking at the page, not a property of one page.
        # Each page gets a fresh item, so carry the state onto it.
        lettering.hidden = self.lettering_hidden
        self.lettering = lettering
        self.fit_page()

    def fit_page(self) -> None:
        if self.scene().sceneRect().isValid():
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)

    def scale_factor(self) -> float:
        return self.transform().m11()

    def is_editing(self) -> bool:
        return self._editing is not None

    # -- interaction ----------------------------------------------------

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        self._reposition_editor()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self._pan_from = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            return

        # Reading a region beats every other drag while the tool is on: that is
        # what the tool is for, and it changes nothing on the page.
        if (
            self.lookup_active
            and event.button() == Qt.LeftButton
            and self._editing is None
            and self.lettering is not None
        ):
            self._drawing_from = self.mapToScene(event.position().toPoint())
            self._drawing_to = self._drawing_from
            return

        # Shift-drag draws a box by hand. Detection is a guess, and where a
        # balloon is genuinely fused with the artwork around it -- black on
        # black -- no guess will ever separate them, so there has to be a way
        # to simply say where the text goes.
        if (
            event.button() == Qt.LeftButton
            and event.modifiers() & Qt.ShiftModifier
            and self._editing is None
            and not self.stamp_active
            and self.lettering is not None
        ):
            self._drawing_from = self.mapToScene(event.position().toPoint())
            self._drawing_to = self._drawing_from
            return

        if self.stamp_active and event.button() == Qt.LeftButton and self.lettering is not None:
            self._stamp_press(self.mapToScene(event.position().toPoint()), event.modifiers())
            return

        if event.button() == Qt.LeftButton and self.lettering is not None:
            point = self.mapToScene(event.position().toPoint())

            # Handles sit on top of everything, including the text, so they can
            # be grabbed where they are drawn -- a corner handle often overlaps
            # the lettering it belongs to.
            if self._editing is None and self._grab_handle(point):
                return

            # While editing, clicks land on the text: place the caret or start
            # a selection drag rather than selecting or moving boxes.
            if self._editing is not None and self._inside_editing_box(point):
                if self._is_triple_click(event.position().toPoint()):
                    self._place_caret(point)
                    self._select_line()
                    self.editor.setFocus()
                    return
                self._place_caret(point, extend=bool(event.modifiers() & Qt.ShiftModifier))
                self._selecting = True
                # A click on the canvas gives the view focus, which would send
                # the next keystroke to the page instead of the text.
                self.editor.setFocus()
                return

            hit = self.lettering.hit_test(point.x(), point.y())

            if self._editing is not None and hit is not self._editing:
                self.commit_edit()

            if hit is not None:
                self.lettering.selected_id = hit.box.id
                self.lettering.update()
                self.selection_changed.emit(hit)
                self.snapshot_requested.emit("Move text", None)
                self._drag_runtime = hit
                self._drag_from = point
                self._drag_origin = (
                    list(hit.box.offset) if hit.box.kind == "bubble" else list(hit.box.rect or [])
                )
                return

            self.lettering.selected_id = None
            self.lettering.update()
            self.selection_changed.emit(None)
            self.box_requested.emit(int(point.x()), int(point.y()))
            return

        super().mousePressEvent(event)

    # -- scale and rotate handles ---------------------------------------

    def _grab_handle(self, point) -> bool:
        """Start a handle drag if the click landed on one."""
        runtime = self.lettering.runtime_for(self.lettering.selected_id)
        if runtime is None or runtime.mask is None:
            return False

        name = self._handle_at(runtime, point)
        if name is None:
            return False

        box = runtime.box
        cx, cy = runtime.centre
        centre = QPointF(cx + box.offset[0], cy + box.offset[1])
        reach = math.hypot(point.x() - centre.x(), point.y() - centre.y())
        if name == "rotate":
            self.snapshot_requested.emit("Rotate text", ("rotate", box.id, 0))
        elif name in EDGES:
            self.snapshot_requested.emit("Resize text area", ("stretch", box.id, 0))
        else:
            self.snapshot_requested.emit("Scale text", ("scale", box.id, 0))

        self._handle_drag = name
        self._handle_runtime = runtime
        self._handle_centre = centre
        self._handle_origin = (
            box.font_size,
            box.angle,
            box.auto_fit,
            max(reach, 1e-6),
            math.degrees(math.atan2(point.y() - centre.y(), point.x() - centre.x())),
            runtime.effective_size,
        )
        # A stretch is measured along the box's own axes, so a rotated box
        # widens across its text rather than across the page.
        area = runtime.text_area_rect()
        local = runtime.to_local(point.x(), point.y())
        self._stretch_origin = (
            box.stretch_x,
            box.stretch_y,
            abs(local[0] - area.center().x()),
            abs(local[1] - area.center().y()),
        )
        return True

    def _update_handle_cursor(self, point) -> None:
        runtime = self.lettering.runtime_for(self.lettering.selected_id)
        name = None
        if runtime is not None and runtime.mask is not None:
            name = self._handle_at(runtime, point)

        if name == self._hover_handle:
            return
        self._hover_handle = name
        if name is None:
            self.unsetCursor()
        elif name in ("left", "right"):
            self.setCursor(Qt.SizeHorCursor if not runtime.box.angle else Qt.SizeAllCursor)
        elif name in ("top", "bottom"):
            self.setCursor(Qt.SizeVerCursor if not runtime.box.angle else Qt.SizeAllCursor)
        elif name == "rotate":
            # No stock rotate cursor exists; the open hand is the closest thing
            # that reads as "take hold of this and turn it".
            self.setCursor(Qt.OpenHandCursor)
        else:
            # Which diagonal depends on the corner, and on how far the box has
            # been turned -- past 45 degrees the corners have swapped places.
            angle = (runtime.box.angle + (45 if name in ("tr", "bl") else 0)) % 180
            self.setCursor(
                Qt.SizeBDiagCursor if 22.5 <= angle < 112.5 else Qt.SizeFDiagCursor
            )

    def _handle_at(self, runtime: BoxRuntime, point) -> str | None:
        """Which handle, if any, is under a page point."""
        scale = self.scale_factor() or 1.0
        points = self.lettering.handle_points(runtime, scale)
        if not points:
            return None
        reach = HANDLE_GRAB_SIZE / scale / 2.0
        for name in ("rotate",) + CORNERS + EDGES:  # rotate wins where they overlap
            spot = points[name]
            if abs(point.x() - spot.x()) <= reach and abs(point.y() - spot.y()) <= reach:
                return name
        return None

    def _drag_handle(self, point, modifiers) -> None:
        runtime = self._handle_runtime
        box = runtime.box
        centre = self._handle_centre
        start_size, start_angle, was_fitted, start_reach, start_bearing, fitted = (
            self._handle_origin
        )

        if self._handle_drag in EDGES:
            # Freeze the size, the same way a corner drag does. Left fitting,
            # widening the area only lets the font grow to match and the words
            # break in exactly the same places -- which is the opposite of why
            # anybody drags a side. Held still, the extra room does what it
            # looks like it should and pulls a word back up a line.
            if was_fitted:
                box.font_size = round(fitted, 1)
                box.auto_fit = False

            local = runtime.to_local(point.x(), point.y())
            area = runtime.text_area_rect()
            start_x, start_y, from_x, from_y = self._stretch_origin
            if self._handle_drag in ("left", "right"):
                reach = abs(local[0] - area.center().x())
                box.stretch_x = round(
                    min(max(start_x + reach - from_x, -400.0), 400.0), 1
                )
            else:
                reach = abs(local[1] - area.center().y())
                box.stretch_y = round(
                    min(max(start_y + reach - from_y, -400.0), 400.0), 1
                )
            runtime.relayout(force=True)
            self.lettering.update()
            self.handle_dragged.emit(runtime)
            return

        if self._handle_drag == "rotate":
            bearing = math.degrees(
                math.atan2(point.y() - centre.y(), point.x() - centre.x())
            )
            angle = start_angle + (bearing - start_bearing)
            if modifiers & Qt.ShiftModifier:
                angle = round(angle / 15.0) * 15.0
            # Keep it in the range the inspector's spin box shows.
            box.angle = (angle + 180.0) % 360.0 - 180.0
        else:
            reach = math.hypot(point.x() - centre.x(), point.y() - centre.y())
            # Scaling a balloon cannot change the balloon -- its shape comes
            # from the artwork -- so a corner drag sizes the lettering, which is
            # the thing you are actually trying to make bigger.
            if was_fitted:
                # Start from the size the fit had chosen rather than from a
                # stored one that was never on screen.
                start_size = fitted
                box.auto_fit = False
            base = max(start_size, 1e-6) * (reach / start_reach)
            box.font_size = round(min(max(base, 4.0), 400.0), 1)

        runtime.relayout(force=True)
        self.lettering.update()
        self.handle_dragged.emit(runtime)

    def contextMenuEvent(self, event) -> None:
        """Right-click asks what can be done here.

        Reading a balloon's original text had been a keystroke and nothing else,
        which is no way to find out a feature exists.
        """
        if self.lettering is None:
            super().contextMenuEvent(event)
            return
        point = self.mapToScene(event.pos())
        self.menu_wanted.emit(
            event.globalPos(), self.lettering.hit_test(point.x(), point.y())
        )

    def mouseMoveEvent(self, event) -> None:
        if self._handle_drag is not None:
            self._drag_handle(self.mapToScene(event.position().toPoint()), event.modifiers())
            return

        if self._drawing_from is not None:
            self._drawing_to = self.mapToScene(event.position().toPoint())
            self.viewport().update()
            return

        if self.stamp_active:
            self._cursor_point = self.mapToScene(event.position().toPoint())
            if self._stamp_backup is not None:
                self._stamp_drag(self._cursor_point)
            self.viewport().update()
            if self._stamp_backup is not None:
                return

        if self._selecting:
            self._place_caret(self.mapToScene(event.position().toPoint()), extend=True)
            return

        # Say what a handle will do before it is grabbed.
        if (
            self._pan_from is None
            and self._drag_runtime is None
            and self._editing is None
            and not self.stamp_active
            and self.lettering is not None
        ):
            self._update_handle_cursor(self.mapToScene(event.position().toPoint()))

        if self._pan_from is not None:
            delta = event.position().toPoint() - self._pan_from
            self._pan_from = event.position().toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._reposition_editor()
            return

        if self._drag_runtime is not None and self._drag_from is not None:
            point = self.mapToScene(event.position().toPoint())
            dx = point.x() - self._drag_from.x()
            dy = point.y() - self._drag_from.y()
            box = self._drag_runtime.box
            if box.kind == "bubble":
                box.offset = [self._drag_origin[0] + dx, self._drag_origin[1] + dy]
            else:
                box.rect = [
                    self._drag_origin[0] + dx,
                    self._drag_origin[1] + dy,
                    self._drag_origin[2],
                    self._drag_origin[3],
                ]
                self._drag_runtime.rebuild()
            self.lettering.update()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._handle_drag is not None:
            runtime = self._handle_runtime
            self._handle_drag = None
            self._handle_runtime = None
            runtime.ensure_repair()
            self.lettering.update()
            self.editing_finished.emit()  # one drag is one undo step
            self.text_edited.emit()
            return

        if self._drawing_from is not None:
            start, end = self._drawing_from, self._drawing_to
            self._drawing_from = self._drawing_to = None
            self.viewport().update()
            if self.lookup_active:
                if start is not None and end is not None:
                    x0, x1 = sorted((start.x(), end.x()))
                    y0, y1 = sorted((start.y(), end.y()))
                    if x1 - x0 >= 6 and y1 - y0 >= 6:
                        self.text_wanted.emit(
                            (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
                        )
                return
            if start is not None and end is not None:
                x0, x1 = sorted((start.x(), end.x()))
                y0, y1 = sorted((start.y(), end.y()))
                if x1 - x0 >= 12 and y1 - y0 >= 12:
                    self.box_drawn.emit(int(x0), int(y0), int(x1 - x0), int(y1 - y0))
                else:
                    # A shift-click rather than a drag: a default-sized box.
                    self.box_drawn.emit(
                        int(start.x()) - 210, int(start.y()) - 70, 420, 140
                    )
            return

        if self._stamp_backup is not None:
            backup, self._stamp_backup = self._stamp_backup, None
            self._stamp_last = None
            if backup:
                self.stroke_finished.emit(backup)
            return

        if self._selecting:
            self._selecting = False
            return
        if event.button() == Qt.MiddleButton:
            self._pan_from = None
            self.unsetCursor()
            return
        if self._drag_runtime is not None:
            self._drag_runtime.ensure_repair()
            self._drag_runtime = None
            self._drag_from = None
            if self.lettering:
                self.lettering.update()
            self.text_edited.emit()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.lettering is not None:
            point = self.mapToScene(event.position().toPoint())

            # Double-clicking text you are already editing selects a word.
            if self._editing is not None and self._inside_editing_box(point):
                self._place_caret(point)
                cursor = self.editor.textCursor()
                cursor.select(QTextCursor.WordUnderCursor)
                self.editor.setTextCursor(cursor)
                self._selecting = False
                self._last_double_click = (time.monotonic(), event.position().toPoint())
                self.editor.setFocus()
                return

            hit = self.lettering.hit_test(point.x(), point.y())
            if hit is not None:
                self.start_edit(hit)
                return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:
        if self.stamp_active and event.key() in (Qt.Key_BracketLeft, Qt.Key_BracketRight):
            step = -4 if event.key() == Qt.Key_BracketLeft else 4
            self.stamp_radius = max(1, min(150, self.stamp_radius + step))
            self.brush_resized.emit(self.stamp_radius * 2)
            self.viewport().update()
            return

        # H shows the page as it arrived, so a translation can be checked
        # against the original without undoing the work to see it. Only when not
        # editing -- while typing, h is the letter h, which the editor gets
        # first anyway.
        if (
            event.key() == Qt.Key_H
            and self._editing is None
            and not event.modifiers() & (Qt.ControlModifier | Qt.AltModifier)
        ):
            self.set_lettering_hidden(not self.lettering_hidden)
            return

        # Ctrl+Shift+C reads the balloon rather than copying the translation,
        # so plain Ctrl+C still belongs to the editor.
        if (
            event.key() == Qt.Key_C
            and event.modifiers() & Qt.ControlModifier
            and event.modifiers() & Qt.ShiftModifier
            and self.lettering is not None
        ):
            runtime = self.lettering.runtime_for(self.lettering.selected_id)
            if runtime is not None:
                self.text_wanted.emit(runtime)
            return

        if event.key() == Qt.Key_Escape:
            # First Escape leaves editing but keeps the box selected so you can
            # still adjust it; a second clears the selection for a clean look at
            # the page.
            if self._editing is not None:
                self.commit_edit()
            elif self.lettering is not None and self.lettering.selected_id is not None:
                self.lettering.selected_id = None
                self.lettering.update()
                self.selection_changed.emit(None)
            return

        # Arrow keys nudge the selected box while not editing. Fine positioning
        # by dragging is fiddly, and this is the common case.
        nudges = {
            Qt.Key_Left: (-1, 0),
            Qt.Key_Right: (1, 0),
            Qt.Key_Up: (0, -1),
            Qt.Key_Down: (0, 1),
        }
        if self._editing is None and event.key() in nudges and self.lettering is not None:
            runtime = self.lettering.runtime_for(self.lettering.selected_id)
            if runtime is not None:
                step = 20 if event.modifiers() & Qt.ShiftModifier else 4
                dx, dy = nudges[event.key()]
                box = runtime.box
                self.snapshot_requested.emit("Move text", ("nudge", box.id))
                if box.kind == "bubble":
                    box.offset = [box.offset[0] + dx * step, box.offset[1] + dy * step]
                elif box.rect:
                    box.rect[0] += dx * step
                    box.rect[1] += dy * step
                    runtime.rebuild()
                self.lettering.update()
                self.text_edited.emit()
                return

        super().keyPressEvent(event)

    def set_lettering_hidden(self, hidden: bool) -> None:
        """Show or hide everything the lettering layer draws."""
        hidden = bool(hidden)
        if hidden == self.lettering_hidden:
            return
        self.lettering_hidden = hidden
        if self.lettering is not None:
            self.lettering.hidden = hidden
            self.lettering.update()
        self.lettering_hidden_changed.emit(hidden)

    # -- inline editing -------------------------------------------------

    def start_edit(self, runtime: BoxRuntime) -> None:
        """Open the inline editor over a box.

        The editor is a real text widget, so caret movement, selection and IME
        for non-Latin scripts all behave as they should. Shape-fitted wrapping
        resumes the moment editing ends.
        """
        self.commit_edit()
        # Editing what cannot be seen is typing at nothing, so opening the
        # editor brings the lettering back.
        self.set_lettering_hidden(False)
        self.snapshot_requested.emit("Edit text", ("text", runtime.box.id))
        self._editing = runtime
        self.editor.blockSignals(True)
        self.editor.setPlainText(runtime.box.text)
        self.editor.blockSignals(False)
        self._reposition_editor()
        self.editor.show()
        self.editor.setFocus()
        self.editor.moveCursor(QTextCursor.End)
        self._caret_visible = True
        self._caret_timer.start()
        self._sync_edit_state()

    def _reposition_editor(self) -> None:
        if self._editing is None:
            return
        x, y, w, h = self._editing.bounds
        if not w or not h:
            return

        # The widget draws nothing. It is positioned over the box only so that
        # an IME's candidate window pops up next to the text being composed,
        # and sized in the box's own font so that window is scaled sensibly.
        rect = self.mapFromScene(QRectF(x, y, w, h)).boundingRect()
        self._editor_base_rect = QRect(rect)
        self.editor.setGeometry(rect)

        font = build_font(self._editing.box, max(6.0, self._editing.effective_size))
        font.setPointSizeF(max(6.0, self._editing.effective_size * self.scale_factor()))
        self.editor.setFont(font)

    # -- caret on the real text -----------------------------------------

    def _blink_caret(self) -> None:
        self._caret_visible = not self._caret_visible
        # Only the caret changed, so only the box it sits in needs redrawing.
        # A whole-page repaint twice a second is work the page is doing while
        # you are not even touching it, and it grows with every box you letter.
        self._sync_edit_state(area=self._editing)

    def _sync_edit_state(self, area: BoxRuntime | None = None) -> None:
        """Publish cursor and selection to the canvas so it can draw them.

        `area` limits the repaint to one box; leave it off whenever the change
        could affect anything else on the page.
        """
        if self.lettering is None:
            return
        if self._editing is None:
            self.lettering.edit = None
        else:
            cursor = self.editor.textCursor()
            self.lettering.edit = (
                self._editing,
                cursor.selectionStart(),
                cursor.selectionEnd(),
                cursor.position(),
                self._caret_visible,
            )
        if area is not None and area.mask is not None:
            self.lettering.update(area.painted_rect())
        else:
            self.lettering.update()

    # -- clone stamp ----------------------------------------------------

    def _stamp_press(self, point, modifiers) -> None:
        layer = self.lettering.repair if self.lettering else None
        if layer is None:
            return
        position = (int(round(point.x())), int(round(point.y())))

        if modifiers & Qt.AltModifier:
            self.stamp_source = position
            self.stamp_offset = None  # re-anchor on the next stroke
            self.viewport().update()
            return

        if self.stamp_source is None:
            return

        if self.stamp_offset is None:
            # Anchor on first paint, so the source keeps its distance from the
            # brush for the rest of the session -- the usual aligned behaviour.
            offset_x = self.stamp_source[0] - position[0]
            offset_y = self.stamp_source[1] - position[1]

            # On screentone, an offset that is not a whole number of lattice
            # periods pastes the dots out of step and leaves a seam no amount of
            # careful painting will hide. Rounding the offset to the lattice
            # makes the copy land dot-on-dot.
            if self.stamp_snap:
                lattice = lattice_at(layer.grey, *self.stamp_source)
                if lattice is not None:
                    period_x, period_y = lattice
                    offset_x = int(round(offset_x / period_x) * period_x)
                    offset_y = int(round(offset_y / period_y) * period_y)
                    self.snapped_to.emit(period_x, period_y)
            self.stamp_offset = (offset_x, offset_y)

        self._stamp_backup = {}
        self._stamp_last = position
        layer.dab(
            position[0],
            position[1],
            position[0] + self.stamp_offset[0],
            position[1] + self.stamp_offset[1],
            self.stamp_radius,
            backup=self._stamp_backup,
        )
        self._after_stamp()

    def _stamp_drag(self, point) -> None:
        layer = self.lettering.repair if self.lettering else None
        if layer is None or self._stamp_last is None or self.stamp_offset is None:
            return
        position = (int(round(point.x())), int(round(point.y())))
        layer.stroke(
            self._stamp_last,
            position,
            self.stamp_offset,
            self.stamp_radius,
            backup=self._stamp_backup,
        )
        self._stamp_last = position
        self._after_stamp()

    def _after_stamp(self) -> None:
        self.lettering.refresh_repair()
        self.lettering.update()

    def drawForeground(self, painter: QPainter, rect) -> None:
        """Brush outline and clone source, drawn over the page."""
        super().drawForeground(painter, rect)

        if self._drawing_from is not None and self._drawing_to is not None:
            painter.save()
            painter.setBrush(HANDLE_TINT)
            painter.setPen(QPen(QColor(64, 132, 255), 0, Qt.SolidLine))
            painter.drawRect(QRectF(self._drawing_from, self._drawing_to).normalized())
            painter.restore()

        if not self.stamp_active:
            return

        painter.save()
        painter.setBrush(Qt.NoBrush)
        if self.stamp_source is not None:
            offset = self.stamp_offset or (0, 0)
            sx, sy = self.stamp_source
            if self._cursor_point is not None and self.stamp_offset is not None:
                sx = self._cursor_point.x() + offset[0]
                sy = self._cursor_point.y() + offset[1]
            painter.setPen(QPen(QColor(40, 170, 90), 0, Qt.DashLine))
            painter.drawEllipse(
                QRectF(sx - self.stamp_radius, sy - self.stamp_radius,
                       self.stamp_radius * 2, self.stamp_radius * 2)
            )

        if self._cursor_point is not None:
            painter.setPen(QPen(QColor(64, 132, 255), 0, Qt.SolidLine))
            painter.drawEllipse(
                QRectF(
                    self._cursor_point.x() - self.stamp_radius,
                    self._cursor_point.y() - self.stamp_radius,
                    self.stamp_radius * 2,
                    self.stamp_radius * 2,
                )
            )
        painter.restore()

    def _is_triple_click(self, position) -> bool:
        """A third click straight after a double one, in the same place.

        Qt has no triple-click event, so it is inferred. Without this the third
        click just collapses the word selection the second one made.
        """
        when, where = self._last_double_click
        if where is None:
            return False
        if (time.monotonic() - when) * 1000.0 > QApplication.doubleClickInterval():
            return False
        return (position - where).manhattanLength() <= 20

    def _select_line(self) -> None:
        """Select the visible line the caret is on, not the whole paragraph."""
        if self._editing is None or not self._editing.lines:
            return
        lines = self._editing.lines
        line = lines[line_for_index(lines, self.editor.textCursor().position())]
        cursor = self.editor.textCursor()
        cursor.setPosition(line.start)
        cursor.setPosition(line.start + len(line.text), QTextCursor.KeepAnchor)
        self._last_double_click = (0.0, None)
        self.editor.setTextCursor(cursor)

    def _editing_metrics(self) -> QFontMetricsF:
        return QFontMetricsF(build_font(self._editing.box, self._editing.effective_size))

    def _index_at(self, scene_point) -> int | None:
        """Source offset nearest a point on the page, for placing the caret."""
        if self._editing is None:
            return None
        local_x, local_y = self._editing.to_local(scene_point.x(), scene_point.y())
        metrics = self._editing_metrics()
        if self._editing.box.vertical:
            column = metrics.height() * self._editing.box.line_spacing
            return index_at_point_vertical(
                self._editing.lines, metrics, local_x, local_y, column
            )
        return index_at_point(self._editing.lines, metrics, local_x, local_y)

    def _inside_editing_box(self, scene_point) -> bool:
        if self._editing is None or self._editing.mask is None:
            return False
        x, y, w, h = self._editing.bounds
        dx, dy = self._editing.box.offset
        # Generous margin: text can spill past the balloon it belongs to.
        return QRectF(x + dx, y + dy, w, h).adjusted(-w * 0.5, -h * 0.5, w * 0.5, h * 0.5).contains(
            scene_point
        )

    def _place_caret(self, scene_point, extend: bool = False) -> None:
        index = self._index_at(scene_point)
        if index is None:
            return
        cursor = self.editor.textCursor()
        cursor.setPosition(index, QTextCursor.KeepAnchor if extend else QTextCursor.MoveAnchor)
        self._caret_visible = True
        self.editor.setTextCursor(cursor)

    # -- caret movement over the rendered lines --------------------------
    #
    # The editor is only an input sink; its own idea of where the lines break
    # comes from a plain rectangle and does not match the shape-fitted wrapping
    # on screen. Left to itself, Up and Down walk invisible lines -- sometimes
    # none at all, if the text happens to fit its box on one. So vertical and
    # line-wise movement is resolved against the laid-out lines instead.

    def eventFilter(self, watched, event):
        if watched is self.editor and event.type() == QEvent.KeyPress:
            if self._handle_edit_key(event):
                return True
        return super().eventFilter(watched, event)

    def _vertical_move(self, key, lines, metrics, index: int, current: int):
        """Where the caret goes for one key press in a vertical column."""
        line = lines[current]

        if key == Qt.Key_Up:
            return index - 1
        if key == Qt.Key_Down:
            return index + 1
        if key == Qt.Key_Home:
            return line.start
        if key == Qt.Key_End:
            return line.start + len(line.text)

        step = 1 if key in (Qt.Key_Left, Qt.Key_PageDown) else -1
        neighbour = current + step
        if neighbour < 0:
            return 0
        if neighbour >= len(lines):
            return len(self._editing.box.text)

        # Keep the position down the column when changing column.
        depth = 0
        for position, glyph in enumerate(line.glyphs):
            if glyph.start >= index:
                break
            depth = position + 1
        other = lines[neighbour]
        if not other.glyphs:
            return other.start
        depth = min(depth, len(other.glyphs))
        if depth >= len(other.glyphs):
            return other.glyphs[-1].start + 1
        return other.glyphs[depth].start

    def _handle_edit_key(self, event) -> bool:
        if self._editing is None or not self._editing.lines:
            return False

        key = event.key()
        if key in (Qt.Key_Tab, Qt.Key_Backtab):
            return True  # swallow: a tab in a speech balloon means nothing
        if self._editing.box.vertical and key in (Qt.Key_Left, Qt.Key_Right):
            pass  # vertical needs these too; handled below
        elif key not in (
            Qt.Key_Up,
            Qt.Key_Down,
            Qt.Key_Home,
            Qt.Key_End,
            Qt.Key_PageUp,
            Qt.Key_PageDown,
        ):
            return False
        if event.modifiers() & Qt.ControlModifier:
            return False  # Ctrl+Home/End mean the whole document; let Qt do it

        lines = self._editing.lines
        metrics = self._editing_metrics()
        cursor = self.editor.textCursor()
        index = cursor.position()
        current = line_for_index(lines, index)

        if self._editing.box.vertical:
            # Reading runs down a column, so Up and Down step character by
            # character and Left and Right move between columns -- Left going
            # forward, because columns advance leftwards.
            target = self._vertical_move(key, lines, metrics, index, current)
            if target is None:
                return False
            mode = (
                QTextCursor.KeepAnchor
                if event.modifiers() & Qt.ShiftModifier
                else QTextCursor.MoveAnchor
            )
            cursor.setPosition(max(0, min(target, len(self._editing.box.text))), mode)
            self._caret_visible = True
            self.editor.setTextCursor(cursor)
            return True

        if key in (Qt.Key_Home, Qt.Key_End):
            line = lines[current]
            target = line.start if key == Qt.Key_Home else line.start + len(line.text)
        else:
            step = {
                Qt.Key_Up: -1,
                Qt.Key_Down: 1,
                Qt.Key_PageUp: -5,
                Qt.Key_PageDown: 5,
            }[key]
            neighbour = current + step
            # A page key that overshoots should land on the first or last line
            # rather than jumping to the very start or end of the text.
            if step in (-5, 5) and 0 <= current < len(lines):
                neighbour = max(0, min(len(lines) - 1, neighbour))
            if neighbour < 0:
                target = 0
            elif neighbour >= len(lines):
                target = len(self._editing.box.text)
            else:
                # Keep the horizontal position while changing row.
                x = caret_x(lines[current], metrics, index)
                y = lines[neighbour].baseline - metrics.ascent() / 2.0
                found = index_at_point(lines, metrics, x, y)
                if found is None:
                    return False
                target = found

        mode = (
            QTextCursor.KeepAnchor
            if event.modifiers() & Qt.ShiftModifier
            else QTextCursor.MoveAnchor
        )
        cursor.setPosition(max(0, min(target, len(self._editing.box.text))), mode)
        self._caret_visible = True
        self.editor.setTextCursor(cursor)
        return True


    def refresh_editor(self) -> None:
        """Re-apply box styling to the open editor, if there is one."""
        if self._editing is not None:
            self._reposition_editor()

    def _on_editor_changed(self) -> None:
        if self._editing is None:
            return
        self._editing.box.text = self.editor.toPlainText()
        self._editing.box.touched = True  # a real edit: erasing may now apply
        self._editing.relayout()
        self._caret_visible = True
        self._sync_edit_state()
        if self.lettering:
            self.lettering.update()
        self.text_edited.emit()

    def commit_edit(self) -> None:
        if self._editing is None:
            return
        self._editing.box.text = self.editor.toPlainText()
        self._editing.relayout()
        self._editing = None
        self._selecting = False
        self._caret_timer.stop()
        # Close the undo run, so the next editing session is its own step
        # rather than coalescing back into this one.
        self.editing_finished.emit()
        if self.lettering is not None:
            self.lettering.edit = None
        self.editor.hide()
        if self.lettering:
            self.lettering.update()
        self.setFocus()
        self.text_edited.emit()


class Inspector(QWidget):
    """Per-box properties."""

    changed = Signal()
    snapshot_requested = Signal(str, object)  # label, coalesce key -- before changing
    editing_finished = Signal()  # ends the current undo run
    stroke_finished = Signal(object)  # tile backup from one clone-stamp stroke

    def __init__(self):
        super().__init__()
        self.runtime: BoxRuntime | None = None
        self._loading = False
        self._shown_font = ""

        form = QFormLayout(self)
        form.setLabelAlignment(Qt.AlignRight)

        self.font_box = QComboBox()
        families = QFontDatabase.families()
        self.font_box.addItems(families)
        form.addRow("Font", self.font_box)

        self.default_font_button = QPushButton("Letter in this by default")
        form.addRow("", self.default_font_button)

        self.load_font_button = QPushButton("Load font file…")
        form.addRow("", self.load_font_button)

        self.auto_fit = QCheckBox("Fit to balloon")
        form.addRow("", self.auto_fit)

        self.size_box = QDoubleSpinBox()
        self.size_box.setRange(4, 400)
        self.size_box.setDecimals(1)
        form.addRow("Size", self.size_box)

        self.max_lines_box = QSpinBox()
        self.max_lines_box.setRange(0, 20)
        self.max_lines_box.setSpecialValueText("as it fits")
        self.max_lines_box.setToolTip(
            "Most lines the text may break onto.\n"
            "Fitting picks the biggest font that fits, and stacking words almost\n"
            "always allows a bigger one, so a wide caption tends to come out as\n"
            "two lines down the middle. Set 1 to keep it on one line across.\n"
            "Ignored if no size can honour it."
        )
        form.addRow("Max lines", self.max_lines_box)

        self.spacing_box = QDoubleSpinBox()
        self.spacing_box.setRange(0.5, 3.0)
        self.spacing_box.setSingleStep(0.05)
        self.spacing_box.setDecimals(2)
        form.addRow("Line spacing", self.spacing_box)

        self.vertical_box = QCheckBox("Vertical (tategaki)")
        self.vertical_box.setToolTip(
            "Columns top-to-bottom, right-to-left, as Japanese is normally set"
        )
        form.addRow("", self.vertical_box)

        self.align_box = QComboBox()
        self.align_box.addItems(["left", "center", "right"])
        form.addRow("Align", self.align_box)

        self.angle_box = QDoubleSpinBox()
        self.angle_box.setRange(-180, 180)
        self.angle_box.setSingleStep(1)
        self.angle_box.setSuffix("°")
        form.addRow("Rotation", self.angle_box)

        self.padding_box = QDoubleSpinBox()
        self.padding_box.setRange(0, 200)
        form.addRow("Inset", self.padding_box)

        stretch = QHBoxLayout()
        stretch.setContentsMargins(0, 0, 0, 0)
        stretch.setSpacing(4)
        self.stretch_x_box = QDoubleSpinBox()
        self.stretch_x_box.setRange(-400, 400)
        self.stretch_x_box.setPrefix("W")
        self.stretch_y_box = QDoubleSpinBox()
        self.stretch_y_box.setRange(-400, 400)
        self.stretch_y_box.setPrefix("H")
        for spin in (self.stretch_x_box, self.stretch_y_box):
            spin.setDecimals(0)
            # Two of these share one row in a panel 300px wide. Left to their
            # own devices they get squeezed to about half what they need and
            # show arrows with no number beside them.
            spin.setMinimumWidth(90)
            spin.setToolTip(
                "Widen or heighten the area text flows into, in pixels.\n"
                "The balloon a letterer wants to fill is often a few pixels\n"
                "bigger than the one the artwork draws, and one word falling\n"
                "to a line of its own is the whole difference.\n"
                "Drag the round side handles to do the same by eye.\n"
                "The erase still follows the balloon, so this cannot paint\n"
                "outside it."
            )
            stretch.addWidget(spin)
        holder_stretch = QWidget()
        holder_stretch.setLayout(stretch)
        form.addRow("Text area", holder_stretch)

        # Each edge of the erased area, on its own. Covering the original and
        # placing the translation are different jobs: original lettering can sit
        # a little outside the balloon, and pushing one edge out beats erasing a
        # block of it by hand.
        self.erase_edge_boxes = {}
        for label, pair in (("Erase L R", ("erase_left", "erase_right")),
                            ("Erase T B", ("erase_top", "erase_bottom"))):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            for field in pair:
                spin = QDoubleSpinBox()
                spin.setRange(-400, 400)
                spin.setDecimals(0)
                spin.setMinimumWidth(90)
                spin.setPrefix(field.rsplit("_", 1)[1][0].upper())
                spin.setToolTip(
                    "Move one edge of the erased area, in pixels.\n"
                    "Positive pushes it outward, negative pulls it in.\n\n"
                    "This may take the erase outside the balloon, which is the\n"
                    "point — original lettering does not always sit inside what\n"
                    "detection found. It will paint over artwork if pushed too far."
                )
                self.erase_edge_boxes[field] = spin
                row.addWidget(spin)
            holder = QWidget()
            holder.setLayout(row)
            form.addRow(label, holder)

        # Their own row: the boxes above already need the whole width, and a
        # third widget beside them is what squeezed the numbers out of view.
        resets = QHBoxLayout()
        resets.setContentsMargins(0, 0, 0, 0)
        resets.setSpacing(4)
        self.reset_stretch = QPushButton("Reset area")
        self.reset_stretch.setToolTip("Back to the shape the artwork gives")
        self.reset_erase = QPushButton("Reset erase")
        self.reset_erase.setToolTip("Put every erase edge back on the balloon")
        resets.addWidget(self.reset_stretch)
        resets.addWidget(self.reset_erase)
        holder_resets = QWidget()
        holder_resets.setLayout(resets)
        form.addRow("", holder_resets)

        colours = QHBoxLayout()
        self.black_button = QPushButton("Black")
        self.white_button = QPushButton("White")
        self.black_button.setCheckable(True)
        self.white_button.setCheckable(True)
        colours.addWidget(self.black_button)
        colours.addWidget(self.white_button)
        holder = QWidget()
        holder.setLayout(colours)
        form.addRow("Text", holder)

        self.outline_slider = QSlider(Qt.Horizontal)
        self.outline_slider.setRange(0, 60)
        form.addRow("Outline", self.outline_slider)

        self.outline_label = QLabel("0 px")
        form.addRow("", self.outline_label)

        self.outline_colour = QComboBox()
        self.outline_colour.addItems(["white", "black"])
        form.addRow("Outline colour", self.outline_colour)

        self.erase_box = QCheckBox("Erase original lettering")
        form.addRow("", self.erase_box)

        self.erase_mode = QComboBox()
        self.erase_mode.addItem("White fill (inside a balloon)", "white")
        self.erase_mode.addItem("Rebuild background (over art)", "rebuild")
        form.addRow("Method", self.erase_mode)

        self.width_box = QDoubleSpinBox()
        self.width_box.setRange(20, 4000)
        self.width_box.setSingleStep(10)
        self.width_row = form.rowCount()
        form.addRow("Box width", self.width_box)

        self.height_box = QDoubleSpinBox()
        self.height_box.setRange(20, 4000)
        self.height_box.setSingleStep(10)
        form.addRow("Box height", self.height_box)

        self.recentre_button = QPushButton("Recentre text")
        form.addRow("", self.recentre_button)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        form.addRow("", self.status)

        self.delete_button = QPushButton("Delete box")
        form.addRow("", self.delete_button)

        self.recentre_button.clicked.connect(self._recentre)

        for widget, signal in (
            (self.font_box, "currentTextChanged"),
            (self.auto_fit, "toggled"),
            (self.size_box, "valueChanged"),
            (self.spacing_box, "valueChanged"),
            (self.max_lines_box, "valueChanged"),
            (self.align_box, "currentTextChanged"),
            (self.vertical_box, "toggled"),
            (self.padding_box, "valueChanged"),
            (self.stretch_x_box, "valueChanged"),
            (self.stretch_y_box, "valueChanged"),
            *((spin, "valueChanged") for spin in self.erase_edge_boxes.values()),
            (self.angle_box, "valueChanged"),
            (self.outline_slider, "valueChanged"),
            (self.outline_colour, "currentTextChanged"),
            (self.erase_box, "toggled"),
            (self.erase_mode, "currentIndexChanged"),
            (self.width_box, "valueChanged"),
            (self.height_box, "valueChanged"),
        ):
            getattr(widget, signal).connect(self._apply)

        self.reset_stretch.clicked.connect(self._reset_stretch)
        self.reset_erase.clicked.connect(self._reset_erase)
        self.black_button.clicked.connect(lambda: self._set_colour("#000000"))
        self.white_button.clicked.connect(lambda: self._set_colour("#FFFFFF"))

        self.setEnabled(False)

    def show_box(self, runtime: BoxRuntime | None) -> None:
        self.runtime = runtime
        self.setEnabled(runtime is not None)
        if runtime is None:
            self.status.setText("")
            return

        box = runtime.box
        self._loading = True
        # The combo shows what the text is actually drawn in, which is not
        # necessarily what the box asked for.
        self._shown_font = available_font(box.font_family)
        self.font_box.setCurrentText(self._shown_font)
        self.auto_fit.setChecked(box.auto_fit)
        self.size_box.setValue(box.font_size)
        self.spacing_box.setValue(box.line_spacing)
        self.max_lines_box.setValue(box.max_lines)
        self.align_box.setCurrentText(box.align)
        self.vertical_box.setChecked(box.vertical)
        self.padding_box.setValue(box.padding)
        self.stretch_x_box.setValue(box.stretch_x)
        self.stretch_y_box.setValue(box.stretch_y)
        for field, spin in self.erase_edge_boxes.items():
            spin.setValue(getattr(box, field))
        self.angle_box.setValue(box.angle)
        self.outline_slider.setValue(int(box.outline_width))
        self.outline_colour.setCurrentText(
            "black" if box.outline_color.upper() == "#000000" else "white"
        )
        self.erase_box.setChecked(box.erase)
        index = self.erase_mode.findData(box.erase_mode)
        self.erase_mode.setCurrentIndex(index if index >= 0 else 0)
        if box.rect:
            self.width_box.setValue(box.rect[2])
            self.height_box.setValue(box.rect[3])
        self._loading = False
        self._refresh_state()

    def _reset_stretch(self) -> None:
        if self.runtime is None:
            return
        self.snapshot_requested.emit("Reset text area", None)
        self.runtime.box.stretch_x = 0.0
        self.runtime.box.stretch_y = 0.0
        self.show_box(self.runtime)
        self.runtime.relayout(force=True)
        self.changed.emit()

    def _reset_erase(self) -> None:
        if self.runtime is None:
            return
        self.snapshot_requested.emit("Reset erase edges", None)
        for field in self.erase_edge_boxes:
            setattr(self.runtime.box, field, 0.0)
        self.show_box(self.runtime)
        self.runtime.refresh_erase()
        self.runtime.ensure_repair()
        self.changed.emit()

    def _recentre(self) -> None:
        if self.runtime is None:
            return
        self.snapshot_requested.emit("Recentre text", None)
        self.runtime.box.offset = [0.0, 0.0]
        self._refresh_state()
        self.changed.emit()

    def _set_colour(self, colour: str) -> None:
        if self.runtime is None:
            return
        self.snapshot_requested.emit("Change text colour", None)
        self.runtime.box.color = colour
        self._refresh_state()
        self.changed.emit()

    def _apply(self) -> None:
        if self._loading or self.runtime is None:
            return
        box = self.runtime.box
        mode_changed = box.erase_mode != self.erase_mode.currentData() or (
            box.erase != self.erase_box.isChecked()
        )
        # Keyed on the specific control, so dragging one slider is a single undo
        # step but adjusting size and then spacing stays two.
        self.snapshot_requested.emit("Change text style", ("prop", box.id, id(self.sender())))
        # Only when the family was actually changed. The combo shows the
        # resolved font, so writing it back unconditionally would quietly bake
        # today's fallback into the saved box -- adjust a slider on a page
        # lettered without the real font installed and it would be stuck in the
        # stand-in for good, immune to ever setting the font properly.
        if self.font_box.currentText() != self._shown_font:
            self._shown_font = self.font_box.currentText()
            box.font_family = self._shown_font
        # Turning off the fit hands you the size it had worked out, not the
        # stored one. Taking the fit off is how you say "nearly right, let me
        # adjust it" -- snapping back to whatever the box was created at means
        # dialling it all the way back before you can start.
        if box.auto_fit and not self.auto_fit.isChecked():
            # Clamped through the spin box's own range so the two cannot drift.
            self._loading = True
            self.size_box.setValue(round(self.runtime.effective_size, 1))
            self._loading = False
            box.font_size = self.size_box.value()
        else:
            box.font_size = self.size_box.value()
        box.auto_fit = self.auto_fit.isChecked()
        box.line_spacing = self.spacing_box.value()
        box.max_lines = self.max_lines_box.value()
        box.align = self.align_box.currentText()
        box.vertical = self.vertical_box.isChecked()
        box.padding = self.padding_box.value()
        box.stretch_x = self.stretch_x_box.value()
        box.stretch_y = self.stretch_y_box.value()
        edges_moved = any(
            getattr(box, field) != spin.value()
            for field, spin in self.erase_edge_boxes.items()
        )
        for field, spin in self.erase_edge_boxes.items():
            setattr(box, field, spin.value())
        box.angle = self.angle_box.value()
        box.outline_width = float(self.outline_slider.value())
        box.outline_color = "#000000" if self.outline_colour.currentText() == "black" else "#FFFFFF"
        box.erase = self.erase_box.isChecked()
        box.erase_mode = self.erase_mode.currentData()

        # A free box is sized here; a balloon takes its shape from the artwork.
        resized = False
        if box.kind == "free" and box.rect:
            width, height = self.width_box.value(), self.height_box.value()
            if (width, height) != (box.rect[2], box.rect[3]):
                box.rect[2], box.rect[3] = width, height
                resized = True

        if resized or mode_changed:
            self.runtime.rebuild()
        elif edges_moved:
            # Just the covered area: rebuilding would re-run balloon detection,
            # which costs far more than nudging an edge is worth.
            self.runtime.refresh_erase()
        self.runtime.relayout(force=True)
        self.runtime.ensure_repair()
        self._refresh_state()
        self.changed.emit()

    def _refresh_state(self) -> None:
        if self.runtime is None:
            return
        box = self.runtime.box
        self.size_box.setEnabled(not box.auto_fit)
        # A line cap works by choosing the size, so it means nothing when the
        # size is being set by hand.
        self.max_lines_box.setEnabled(box.auto_fit)
        self.erase_mode.setEnabled(box.erase)
        self.align_box.setEnabled(not box.vertical)  # columns are always centred
        free = box.kind == "free"
        self.width_box.setEnabled(free)
        self.height_box.setEnabled(free)
        self.black_button.setChecked(box.color.upper() == "#000000")
        self.white_button.setChecked(box.color.upper() != "#000000")
        self.outline_label.setText(f"{int(box.outline_width)} px")

        notes = []
        if box.auto_fit:
            notes.append(f"fitted at {self.runtime.effective_size:.1f} pt")
        if self.runtime.overflow:
            notes.append("too big for the balloon — spilling over")
        if box.erase and box.erase_mode == "rebuild" and self.runtime.repair_note:
            notes.append(self.runtime.repair_note)
        if any(box.offset):
            notes.append(f"moved {box.offset[0]:+.0f}, {box.offset[1]:+.0f}")
        if box.angle:
            notes.append(f"rotated {box.angle:+.0f}°")
        if self.runtime.mask is None:
            notes.append("no region found for this box")
        self.status.setText("  ·  ".join(notes))


class MainWindow(QMainWindow):
    def __init__(self, folder: Path):
        super().__init__()
        # The project resolves the path; do the same here so a relative start
        # folder still shows a name rather than nothing.
        self.setWindowTitle(f"EasyLetterFit — {Path(folder).resolve().name}")
        self.resize(1500, 950)

        self.project = Project.load(folder)
        prefs.set_last_folder(self.project.folder)
        self.page_name: str | None = None
        self.gray: np.ndarray | None = None
        self.colour: np.ndarray | None = None
        self.lettering: LetteringItem | None = None
        self.histories: dict[str, History] = {}
        self._thumbnails: list = []
        self._folder_seen: tuple = ()
        self._thumbnail_timer = QTimer(self)
        self._thumbnail_timer.timeout.connect(self._load_thumbnails)
        self.layers: dict[str, RepairLayer] = {}
        self._font_sources: dict[str, str] = {}  # family -> file it came from

        self.pages = QListWidget()
        self.pages.setIconSize(QPixmap(1, 1).size().expandedTo(QPixmap(90, 120).size()))

        # Every piece of lettering on the page, in the order it was added. A
        # small balloon sitting on top of a big one is hard to click, because
        # the big one is what the cursor lands on; picking from a list needs no
        # aim at all.
        self.boxes_list = QListWidget()
        self.boxes_list.setToolTip(
            "Everything lettered on this page.\n"
            "Click to select it, double-click to edit it."
        )

        side = QSplitter(Qt.Vertical)
        side.addWidget(self.pages)
        side.addWidget(self.boxes_list)
        side.setStretchFactor(0, 3)
        side.setStretchFactor(1, 2)
        side.setMaximumWidth(230)

        self.view = PageView()
        self.inspector = Inspector()
        self.inspector.setFixedWidth(300)

        splitter = QSplitter()
        splitter.addWidget(side)
        splitter.addWidget(self.view)
        splitter.addWidget(self.inspector)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self.pages.currentItemChanged.connect(self._page_selected)
        self.pages.itemClicked.connect(self._item_chosen)
        self.pages.itemActivated.connect(self._item_chosen)
        self.boxes_list.itemClicked.connect(self._box_row_chosen)
        self.boxes_list.currentItemChanged.connect(
            lambda current, _previous: self._box_row_chosen(current)
        )
        self.boxes_list.itemDoubleClicked.connect(self._box_row_edit)
        self.view.selection_changed.connect(self.inspector.show_box)
        self.view.selection_changed.connect(self._highlight_box_row)
        self.view.selection_changed.connect(
            lambda rt: self._discard_abandoned(rt.box.id if rt else None)
        )
        self.view.box_requested.connect(self._create_box)
        self.view.box_drawn.connect(self._create_drawn_box)
        self.view.text_edited.connect(lambda: self.inspector.show_box(self.inspector.runtime))
        self.view.text_edited.connect(self.refresh_boxes_list)
        # Dragging a handle moves the size and rotation controls with it, so the
        # panel never disagrees with the page.
        self.view.handle_dragged.connect(self.inspector.show_box)
        self.view.text_wanted.connect(self._copy_original_text)
        self.view.menu_wanted.connect(self._show_canvas_menu)
        self.boxes_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.boxes_list.customContextMenuRequested.connect(self._show_list_menu)
        self.view.snapshot_requested.connect(self._snapshot)
        self.inspector.snapshot_requested.connect(self._snapshot)
        self.view.selection_changed.connect(self._end_coalescing)
        self.view.editing_finished.connect(self._end_coalescing)
        self.view.stroke_finished.connect(self._record_stroke)
        self.view.snapped_to.connect(self._report_snap)
        self.inspector.changed.connect(self._redraw)
        self.inspector.delete_button.clicked.connect(self._delete_box)
        self.inspector.load_font_button.clicked.connect(self._load_font_file)
        self.inspector.default_font_button.clicked.connect(self._use_font_everywhere)

        self._build_toolbar()
        # Before the pages, so the first box created is already in the right
        # font and the inspector opens showing it.
        self._restore_fonts()
        self._load_pages()
        self._warn_about_font()

    # -- setup ----------------------------------------------------------

    def _build_toolbar(self) -> None:
        bar = self.addToolBar("Main")
        bar.setMovable(False)

        save = QAction("Save", self)
        save.setShortcut(QKeySequence.Save)
        save.triggered.connect(self._save)
        bar.addAction(save)

        bar.addSeparator()

        # The three export controls together. The size used to sit at the far
        # end of the bar next to "Snap to tone", where a bare "1x" reads as
        # though the tick is what it belongs to.
        export = QAction("Export page…", self)
        export.triggered.connect(self._export_page)
        bar.addAction(export)

        export_all = QAction("Export all", self)
        export_all.triggered.connect(self._export_all)
        bar.addAction(export_all)

        # A standalone label, not a word like "at" that reads as a qualifier on
        # whichever button happens to sit before it. This applies to both.
        scale_hint = (
            "Upscale both exports by this much.\n\n"
            "The artwork is interpolated — there is no more of it than there\n"
            "was — but the lettering is redrawn at the larger size, so it is\n"
            "genuinely sharper rather than merely bigger."
        )
        scale_label = QLabel("   Upscale: ")
        scale_label.setToolTip(scale_hint)
        bar.addWidget(scale_label)

        self.scale_box = QComboBox()
        for label, factor in (("1x", 1), ("2x", 2), ("4x", 4)):
            self.scale_box.addItem(label, factor)
        self.scale_box.setToolTip(scale_hint)
        remembered = prefs.export_scale()
        index = self.scale_box.findData(remembered)
        if index >= 0:
            self.scale_box.setCurrentIndex(index)
        self.scale_box.currentIndexChanged.connect(
            lambda: prefs.set_export_scale(self.export_scale())
        )
        bar.addWidget(self.scale_box)

        bar.addSeparator()

        new_project = QAction("New project…", self)
        new_project.setShortcut(QKeySequence.New)
        new_project.triggered.connect(self._new_project)
        bar.addAction(new_project)

        add_pages = QAction("Add pages…", self)
        add_pages.triggered.connect(self._add_pages)
        bar.addAction(add_pages)

        refresh = QAction("Refresh", self)
        refresh.setShortcut(QKeySequence.Refresh)
        refresh.triggered.connect(self.refresh_folder)
        bar.addAction(refresh)

        open_folder = QAction("Open folder…", self)
        open_folder.setShortcut(QKeySequence.Open)
        open_folder.triggered.connect(self._choose_folder)
        bar.addAction(open_folder)

        bar.addSeparator()
        self.undo_action = QAction("Undo", self)
        self.undo_action.setShortcut(QKeySequence.Undo)
        self.undo_action.triggered.connect(self._undo)
        bar.addAction(self.undo_action)

        self.redo_action = QAction("Redo", self)
        self.redo_action.setShortcuts([QKeySequence.Redo, QKeySequence("Ctrl+Shift+Z")])
        self.redo_action.triggered.connect(self._redo)
        bar.addAction(self.redo_action)

        bar.addSeparator()
        self.stamp_action = QAction("Clone stamp", self)
        self.stamp_action.setCheckable(True)
        self.stamp_action.toggled.connect(self._toggle_stamp)
        bar.addAction(self.stamp_action)

        self.brush_box = QSpinBox()
        self.brush_box.setRange(2, 300)
        self.brush_box.setValue(48)
        self.brush_box.setSuffix(" px")
        self.brush_box.setToolTip("Brush size")
        self.brush_box.valueChanged.connect(self._set_brush)
        # Connected here rather than with the other signals: the toolbar is
        # built after those, so the widget does not exist yet.
        self.view.brush_resized.connect(self.brush_box.setValue)
        bar.addWidget(self.brush_box)

        self.snap_box = QCheckBox("Snap to tone")
        self.snap_box.setChecked(True)
        self.snap_box.setToolTip(
            "Align the clone offset to the screentone lattice so dots land dot-on-dot"
        )
        self.snap_box.toggled.connect(self._set_snap)
        bar.addWidget(self.snap_box)

        bar.addSeparator()
        self.lookup_action = QAction("Copy text", self)
        self.lookup_action.setCheckable(True)
        self.lookup_action.setToolTip(
            "Turn on, then drag a rectangle over any text to read it onto the\n"
            "clipboard — a single unknown word, or text with no balloon\n"
            "around it.\n\n"
            "To copy a whole balloon instead: right-click it, or select it and\n"
            "press Ctrl+Shift+C."
        )
        self.lookup_action.toggled.connect(self._toggle_lookup)
        bar.addAction(self.lookup_action)

        self.hide_action = QAction("Hide lettering", self)
        self.hide_action.setCheckable(True)
        self.hide_action.setToolTip(
            "Show the page as it arrived, so a translation can be checked "
            "against the original (H)"
        )
        # No shortcut on the action itself: an application shortcut would take
        # the letter h away from the editor. The canvas handles the key, and
        # only when it is not being typed into.
        self.hide_action.toggled.connect(self.view.set_lettering_hidden)
        self.view.lettering_hidden_changed.connect(self._lettering_hidden)
        bar.addAction(self.hide_action)

        fit = QAction("Fit", self)
        fit.triggered.connect(self.view.fit_page)
        bar.addAction(fit)

        self.delete_action = QAction("Delete box", self)
        self.delete_action.setShortcut(QKeySequence.Delete)
        self.delete_action.triggered.connect(self._delete_box)
        self.addAction(self.delete_action)

    def _load_pages(self, keep: str | None = None) -> None:
        """Fill the side panel with this folder's subfolders and pages.

        `keep` names a page to stay on, so a rescan does not throw you back to
        the first page of the chapter.
        """
        self.pages.blockSignals(True)
        self.pages.clear()
        self.pages.blockSignals(False)
        self._thumbnails = []

        style = self.style()
        folder = self.project.folder

        parent = folder.parent
        if parent != folder:
            item = QListWidgetItem("..")
            item.setIcon(style.standardIcon(QStyle.SP_FileDialogToParent))
            item.setData(Qt.UserRole, str(parent))
            item.setData(KIND_ROLE, "folder")
            self.pages.addItem(item)

        for sub in self.project.subfolders():
            try:
                pages = len(Project(sub).image_files())
            except OSError:
                pages = 0
            item = QListWidgetItem(f"{sub.name}  ({pages})" if pages else sub.name)
            item.setIcon(style.standardIcon(QStyle.SP_DirIcon))
            item.setData(Qt.UserRole, str(sub))
            item.setData(KIND_ROLE, "folder")
            self.pages.addItem(item)

        for path in self.project.image_files():
            item = QListWidgetItem(path.name)
            item.setData(Qt.UserRole, str(path))
            item.setData(KIND_ROLE, "page")
            self.pages.addItem(item)
            self._thumbnails.append(item)

        # Thumbnails are decoded a few at a time. A chapter folder can hold
        # hundreds of 600 dpi pages, and loading them all up front would freeze
        # the window for seconds before anything appeared.
        if self._thumbnails:
            self._thumbnail_timer.start(0)

        self._folder_seen = self._folder_signature()

        wanted = None
        first = None
        for row in range(self.pages.count()):
            if self.pages.item(row).data(KIND_ROLE) != "page":
                continue
            if first is None:
                first = row
            if keep is not None and self.pages.item(row).text() == keep:
                wanted = row
                break

        target = wanted if wanted is not None else first
        if target is not None:
            self.pages.setCurrentRow(target)
        else:
            # Nothing to letter here, so say what to do rather than showing a
            # blank canvas with no explanation.
            self.statusBar().showMessage(
                f"No pages in {self.project.folder.name} — "
                "use “Add pages…”, or pick a folder on the left"
            )

    def _load_thumbnails(self) -> None:
        for _ in range(3):
            if not self._thumbnails:
                self._thumbnail_timer.stop()
                return
            item = self._thumbnails.pop(0)
            reader = QImageReader(item.data(Qt.UserRole))
            size = reader.size()
            if size.isValid() and size.width() and size.height():
                scale = min(90 / size.width(), 120 / size.height())
                reader.setScaledSize(
                    QSize(max(1, int(size.width() * scale)), max(1, int(size.height() * scale)))
                )
            image = reader.read()
            if not image.isNull():
                item.setIcon(QPixmap.fromImage(image))

    def _copy_pages(self, files: list[str], destination: Path) -> int:
        """Copy chosen images into a project folder, never overwriting."""
        copied = 0
        for name in files:
            source = Path(name)
            if source.parent == destination:
                continue  # already here
            target = destination / source.name
            attempt = 1
            while target.exists():
                target = destination / f"{source.stem}_{attempt}{source.suffix}"
                attempt += 1
            try:
                shutil.copy2(source, target)
                copied += 1
            except OSError:
                pass
        return copied

    def _new_project(self) -> None:
        """Make a folder beside the current one and open it.

        There is no project file to create: a folder holding pages *is* a
        project, and its sidecar appears the first time something is lettered.
        This exists because making the folder and getting pages into it should
        not require leaving the app.
        """
        name, accepted = QInputDialog.getText(
            self, "New project", f"Folder name, inside {self.project.folder.name}:"
        )
        if not accepted:
            return

        name = name.strip()
        if not name or name in {".", ".."} or any(sep in name for sep in "/\\"):
            QMessageBox.warning(self, "New project", "That is not a usable folder name.")
            return

        destination = self.project.folder / name
        if destination.exists():
            QMessageBox.warning(self, "New project", f"“{name}” already exists here.")
            return

        try:
            destination.mkdir(parents=True)
        except OSError as error:
            QMessageBox.warning(self, "New project", f"Could not create it: {error}")
            return

        files, _ = QFileDialog.getOpenFileNames(
            self, f"Add pages to {name} (optional)", str(self.project.folder), IMAGE_FILTER
        )
        copied = self._copy_pages(files, destination) if files else 0

        self.open_folder(destination)
        self.statusBar().showMessage(
            f"Created {name} with {copied} page(s)"
            if copied
            else f"Created {name} — use “Add pages…” to bring images in"
        )

    def _add_pages(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, f"Add pages to {self.project.folder.name}", str(self.project.folder),
            IMAGE_FILTER,
        )
        if not files:
            return

        copied = self._copy_pages(files, self.project.folder)
        self._load_pages()
        self.statusBar().showMessage(f"Added {copied} page(s)")

    def _folder_signature(self) -> tuple:
        """Cheap fingerprint of what is in this folder right now."""
        try:
            return (
                tuple(p.name for p in self.project.image_files()),
                tuple(p.name for p in self.project.subfolders()),
            )
        except OSError:
            return ()

    def refresh_folder(self) -> None:
        """Rescan the folder, staying on the page you were looking at."""
        self.view.commit_edit()
        self._load_pages(keep=self.page_name)
        self.statusBar().showMessage(f"Rescanned {self.project.folder.name}")

    def changeEvent(self, event) -> None:
        # Folders and images made in Explorer are invisible until something
        # looks again, and coming back to the window is exactly when a user
        # expects to see what they just put there.
        if event.type() == QEvent.ActivationChange and self.isActiveWindow():
            if self._folder_signature() != self._folder_seen:
                self._load_pages(keep=self.page_name)
        super().changeEvent(event)

    def _choose_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Open folder of pages", str(self.project.folder)
        )
        if chosen:
            self.open_folder(Path(chosen))

    def open_folder(self, folder: Path) -> None:
        """Switch to another folder, which is its own separate project."""
        self._save_quietly()
        self.view.commit_edit()

        self.project = Project.load(Path(folder))
        prefs.set_last_folder(self.project.folder)
        # Both are keyed by page name, which is only unique within a folder.
        self.histories.clear()
        self.layers.clear()

        self.page_name = None
        self.gray = None
        self.colour = None
        self.lettering = None
        self.view.lettering = None
        self.view.scene().clear()
        self.inspector.show_box(None)

        self.setWindowTitle(f"EasyLetterFit — {self.project.folder.name}")
        # A font kept with the pages belongs to those pages; pick it up when the
        # folder is opened rather than only at startup.
        found = self._register_folder_fonts()
        self._load_pages()
        self._update_history_actions()
        if found:
            self._refresh_font_list()
            self.statusBar().showMessage(
                f"Found {', '.join(sorted(set(found)))} in this folder"
            )

    def _item_chosen(self, item) -> None:
        if item is not None and item.data(KIND_ROLE) == "folder":
            self.open_folder(Path(item.data(Qt.UserRole)))

    def _warn_about_font(self) -> None:
        """Say which font lettering will come out in, until one is picked.

        Naming a particular font here -- as this once named the one its author
        happened to own -- reads as a requirement, and sends people hunting for
        something they do not need. Nothing is required; this only reports what
        is being used in the absence of a choice.
        """
        if prefs.font_family():
            return  # a font has been chosen; nothing to explain
        self.statusBar().showMessage(
            f"Lettering in {default_font()} for now. To use your own: "
            "“Load font file…”, then “Letter in this by default”."
        )

    # -- fonts ----------------------------------------------------------

    def _restore_fonts(self) -> None:
        """Re-register remembered fonts and adopt the chosen family.

        Qt drops application fonts when the process ends, so a font chosen last
        session has to be loaded again before its name means anything.
        """
        for path in prefs.font_files():
            self._register_font_file(path)
        self._register_folder_fonts()

        family = prefs.font_family()
        if family:
            set_chosen_font(family)
        self._refresh_font_list()

    def _register_font_file(self, path: str) -> list[str]:
        """Register a font file and note which families it provides."""
        font_id = QFontDatabase.addApplicationFont(path)
        if font_id < 0:
            return []
        families = QFontDatabase.applicationFontFamilies(font_id)
        for family in families:
            self._font_sources[family] = path
        return families

    def _register_folder_fonts(self) -> list[str]:
        """Load any font files kept beside the pages, and report new families."""
        found = []
        for path in prefs.fonts_beside(self.project.folder):
            found.extend(self._register_font_file(str(path)))
        return found

    def _refresh_font_list(self) -> None:
        """Repopulate the family list without disturbing the current choice."""
        box = self.inspector.font_box
        current = box.currentText()
        blocked = box.blockSignals(True)
        box.clear()
        box.addItems(QFontDatabase.families())
        if current:
            box.setCurrentText(current)
        box.blockSignals(blocked)
        self._show_chosen_font()

    def _show_chosen_font(self) -> None:
        family = prefs.font_family()
        label = family if family else f"not set — using {default_font()}"
        self.inspector.default_font_button.setToolTip(
            f"Lettering font: {label}.\n"
            "New balloons start in it, and any box asking for a font this "
            "machine does not have\nfalls back to it rather than to a stand-in.\n"
            "Press this to switch to the family selected above."
        )

    # -- pages ----------------------------------------------------------

    def _page_selected(self, current, _previous) -> None:
        if current is None or current.data(KIND_ROLE) != "page":
            return
        self._save_quietly()

        path = Path(current.data(Qt.UserRole))
        try:
            # Both: detection and the tone rebuild read shape and density, but
            # the clone stamp copies what a person pointed at, colour and all.
            colour = load_colour(str(path))
            gray = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
        except OSError as error:
            # Never fail silently here: this runs from a signal, so an exception
            # would vanish and the page would just never appear.
            self.statusBar().showMessage(f"Could not read {path.name}: {error}")
            self.page_name = None
            self.gray = None
            self.colour = None
            self.lettering = None
            self.view.lettering = None
            self.view.scene().clear()
            self.inspector.show_box(None)
            return

        self.page_name = path.name
        self.gray = gray
        self.colour = colour

        pixmap = QPixmap(str(path))
        self.lettering = LetteringItem((pixmap.width(), pixmap.height()))
        self.lettering.runtimes = [
            BoxRuntime(box, self.gray, self.colour) for box in self.project.page(self.page_name).boxes
        ]
        for runtime in self.lettering.runtimes:
            runtime.ensure_repair()
        self.lettering.repair = self._load_repair(self.page_name, self.colour)
        self.lettering.refresh_repair()
        self.view.show_page(pixmap, self.lettering)
        self.inspector.show_box(None)
        self.refresh_boxes_list()
        self._update_history_actions()

    def _create_box(self, x: int, y: int) -> None:
        """Clicking empty page tries for a balloon, else makes a free box."""
        if self.gray is None or self.lettering is None:
            return

        self._snapshot("Add text box")
        box = TextBox(kind="bubble", seed=[x, y], text="", font_family=chosen_font())
        runtime = BoxRuntime(box, self.gray, self.colour)

        if runtime.mask is None:
            side = 420
            box.kind = "free"
            box.seed = None
            box.rect = [x - side / 2, y - side / 6, side, side / 3]
            # No balloon here, so this text sits over artwork: reconstruct the
            # background rather than wiping a white rectangle over the drawing.
            box.erase_mode = "rebuild"
            box.outline_width = 8.0
            self.statusBar().showMessage(
                "No balloon here, so a plain box — " + describe_detection(self.gray, x, y)
            )
            runtime = BoxRuntime(box, self.gray, self.colour)
            if runtime.mask is None:
                return

        _letter_for_background(runtime)
        runtime.ensure_repair()
        self.project.page(self.page_name).boxes.append(box)
        self.lettering.runtimes.append(runtime)
        self.lettering.selected_id = box.id
        self.lettering.update()
        self.inspector.show_box(runtime)
        self.refresh_boxes_list()
        # Selection only. Editing starts on double-click, so a stray click never
        # disturbs the artwork.

    def _create_drawn_box(self, x: int, y: int, width: int, height: int) -> None:
        """Make a plain box exactly where it was drawn."""
        if self.gray is None or self.lettering is None:
            return

        self._snapshot("Add text box")
        box = TextBox(
            kind="free", rect=[x, y, width, height], text="",
            font_family=chosen_font(),
        )
        # White, not a reconstruction. You draw a box by hand exactly where the
        # program failed to make sense of the area on its own -- a balloon fused
        # with the artwork, lettering the detector would not take. Answering
        # that by running another guess over the same area doubles down on the
        # thing that just went wrong, and a wrong rebuild is a patch of somebody
        # else's texture pasted into the page. "Rebuild background" is one
        # choice away in the panel for when it is the right tool.
        box.erase_mode = "white"
        box.outline_width = 8.0

        runtime = BoxRuntime(box, self.gray, self.colour)
        if runtime.mask is None:
            return
        _letter_for_background(runtime)
        runtime.ensure_repair()

        self.project.page(self.page_name).boxes.append(box)
        self.lettering.runtimes.append(runtime)
        self.lettering.selected_id = box.id
        self.lettering.update()
        self.inspector.show_box(runtime)
        self.refresh_boxes_list()
        fill = "black" if runtime.bubble_dark else "white"
        self.statusBar().showMessage(
            f"Drew a {width}×{height} box — clears to {fill}. "
            "Change “Method” to rebuild the artwork instead."
        )

    # -- the list of lettering on this page -----------------------------

    def refresh_boxes_list(self) -> None:
        """Rebuild the list to match the page, keeping the current selection."""
        selected = self.lettering.selected_id if self.lettering else None
        blocked = self.boxes_list.blockSignals(True)
        self.boxes_list.clear()

        runtimes = self.lettering.runtimes if self.lettering else []
        for number, runtime in enumerate(runtimes, 1):
            text = " ".join(runtime.box.text.split())
            if not text:
                text = "(empty)"
            elif len(text) > 40:
                text = text[:39] + "…"
            item = QListWidgetItem(f"{number}. {text}")
            item.setData(Qt.UserRole, runtime.box.id)
            # The whole line, for anything the row is too narrow to show.
            item.setToolTip(runtime.box.text.strip() or "(empty)")
            self.boxes_list.addItem(item)
            if runtime.box.id == selected:
                self.boxes_list.setCurrentItem(item)

        self.boxes_list.blockSignals(blocked)

    def _highlight_box_row(self, runtime) -> None:
        """Follow a selection made on the page itself."""
        box_id = runtime.box.id if runtime is not None else None
        blocked = self.boxes_list.blockSignals(True)
        if box_id is None:
            self.boxes_list.setCurrentRow(-1)
        else:
            for row in range(self.boxes_list.count()):
                item = self.boxes_list.item(row)
                if item.data(Qt.UserRole) == box_id:
                    self.boxes_list.setCurrentItem(item)
                    self.boxes_list.scrollToItem(item)
                    break
        self.boxes_list.blockSignals(blocked)

    def _runtime_from_row(self, item):
        if item is None or self.lettering is None:
            return None
        return self.lettering.runtime_for(item.data(Qt.UserRole))

    def _box_row_chosen(self, item) -> None:
        """Select the box a row stands for, and bring it into view."""
        runtime = self._runtime_from_row(item)
        if runtime is None:
            return

        self.view.commit_edit()
        self.lettering.selected_id = runtime.box.id
        self.lettering.update()
        self.inspector.show_box(runtime)
        # Selecting something you cannot see is not much of a selection. This
        # only scrolls when it has to, so picking a row already on screen does
        # not shift the page under you.
        self.view.ensureVisible(runtime.painted_rect(), 60, 60)
        self._discard_abandoned(runtime.box.id)
        self._end_coalescing()

    def _box_row_edit(self, item) -> None:
        runtime = self._runtime_from_row(item)
        if runtime is not None:
            self.view.ensureVisible(runtime.painted_rect(), 60, 60)
            self.view.start_edit(runtime)

    def _delete_box(self) -> None:
        runtime = self.inspector.runtime
        if runtime is None or self.lettering is None:
            return
        self.view.commit_edit()
        self._snapshot("Delete text box")
        self.lettering.runtimes.remove(runtime)
        boxes = self.project.page(self.page_name).boxes
        if runtime.box in boxes:
            boxes.remove(runtime.box)
        self.lettering.selected_id = None
        self.lettering.update()
        self.inspector.show_box(None)
        self.refresh_boxes_list()

    def _discard_abandoned(self, keep: str | None) -> None:
        """Drop boxes that were created and never typed into.

        Clicking a balloon makes a box; walking away from it without typing
        should leave no trace, or they pile up invisibly and the next click
        selects one of them rather than doing anything.
        """
        if self.lettering is None or self.page_name is None:
            return

        stale = [
            r for r in self.lettering.runtimes
            if r.box.id != keep and not r.box.text.strip() and not r.box.touched
        ]
        if not stale:
            return

        boxes = self.project.page(self.page_name).boxes
        for runtime in stale:
            self.lettering.runtimes.remove(runtime)
            if runtime.box in boxes:
                boxes.remove(runtime.box)
        self.lettering.update()
        self.refresh_boxes_list()

    def _redraw(self) -> None:
        if self.lettering:
            self.lettering.update()
        # Keep an open inline editor in step with the inspector.
        self.view.refresh_editor()

    # -- clone stamp ----------------------------------------------------

    def _toggle_stamp(self, active: bool) -> None:
        if active and self.lookup_action.isChecked():
            self.lookup_action.setChecked(False)
        self.view.commit_edit()
        self.view.stamp_active = active
        self.view.viewport().setCursor(Qt.CrossCursor if active else Qt.ArrowCursor)
        self.view.viewport().update()
        self.statusBar().showMessage(
            "Clone stamp: Alt+click to set the source, then paint over what you want covered."
            if active
            else "Clone stamp off"
        )

    def _report_snap(self, period_x: int, period_y: int) -> None:
        self.statusBar().showMessage(
            f"Clone offset snapped to the {period_x}x{period_y} px screentone lattice"
        )

    def _set_brush(self, size: int) -> None:
        self.view.stamp_radius = max(1, size // 2)
        self.view.viewport().update()

    def _set_snap(self, on: bool) -> None:
        self.view.stamp_snap = on

    def _record_stroke(self, backup: dict) -> None:
        history = self._history()
        if history is not None:
            history.push(RepairEntry("Clone stamp", backup))
        self._update_history_actions()

    def _repair_path(self, page_name: str) -> Path:
        # The page's whole filename, extension included, then ".png" -- so this
        # reads "15_IDWC.png.png". Ugly, but two pages differing only by
        # extension cannot then collide and silently share hand-painted
        # repairs, which are the most expensive thing here to redo.
        return self.project.folder / "repairs" / f"{page_name}.png"

    def _load_repair(self, page_name: str, page: np.ndarray) -> RepairLayer:
        layer = self.layers.get(page_name)
        if layer is not None:
            return layer

        layer = RepairLayer(page)
        path = self._repair_path(page_name)
        if path.exists():
            stored = imread_unicode(str(path))
            if stored is not None and stored.ndim == 3 and stored.shape[2] == 4:
                layer.load_bgra(stored)
        self.layers[page_name] = layer
        return layer

    def _save_repairs(self) -> None:
        for name, layer in self.layers.items():
            if layer.is_empty:
                continue
            path = self._repair_path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            imwrite_unicode(str(path), layer.to_bgra())

    # -- undo -----------------------------------------------------------

    def capture_boxes(self) -> list:
        return [b.to_dict() for b in self.project.page(self.page_name).boxes]

    def restore_boxes(self, data: list) -> None:
        self._apply_snapshot(data)

    def swap_repair(self, tiles: dict) -> None:
        if self.lettering is None or self.lettering.repair is None:
            return
        self.lettering.repair.swap(tiles)
        self.lettering.refresh_repair()
        self.lettering.update()


    def _history(self) -> History | None:
        if self.page_name is None:
            return None
        if self.page_name not in self.histories:
            name = self.page_name
            self.histories[name] = History(
                lambda: [b.to_dict() for b in self.project.page(name).boxes]
            )
        return self.histories[self.page_name]

    def _snapshot(self, label: str, coalesce=None) -> None:
        history = self._history()
        if history is not None:
            history.snapshot(label, coalesce)
        self._update_history_actions()

    def _end_coalescing(self, *_args) -> None:
        history = self._history()
        if history is not None:
            history.break_coalescing()

    def _apply_snapshot(self, boxes_data: list) -> None:
        """Restore a captured set of boxes.

        Runtimes whose region is unchanged are kept and simply re-laid out;
        rebuilding one means re-running balloon detection, which is far too slow
        to do for every box on every undo.
        """
        if self.lettering is None or self.page_name is None:
            return
        self.view.commit_edit()

        def region_of(box: TextBox):
            return (box.kind, tuple(box.seed or ()), tuple(box.rect or ()))

        existing = {r.box.id: r for r in self.lettering.runtimes}
        runtimes = []
        for data in boxes_data:
            box = TextBox.from_dict(data)
            previous = existing.get(box.id)
            if previous is not None and region_of(previous.box) == region_of(box):
                previous.box = box
                previous.relayout(force=True)
                runtimes.append(previous)
            else:
                runtimes.append(BoxRuntime(box, self.gray, self.colour))
        for runtime in runtimes:
            runtime.ensure_repair()

        self.lettering.runtimes = runtimes
        self.project.page(self.page_name).boxes = [r.box for r in runtimes]

        if self.lettering.runtime_for(self.lettering.selected_id) is None:
            self.lettering.selected_id = None
        self.lettering.edit = None
        self.lettering.update()
        self.inspector.show_box(self.lettering.runtime_for(self.lettering.selected_id))
        self.refresh_boxes_list()
        self._update_history_actions()

    def _undo(self) -> None:
        # While typing, Ctrl+Z belongs to the text editor's own undo stack.
        if self.view.is_editing():
            self.view.editor.undo()
            return
        history = self._history()
        label = history.undo(self) if history else None
        self.statusBar().showMessage(f"Undid: {label}" if label else "Nothing to undo")
        self._update_history_actions()

    def _redo(self) -> None:
        if self.view.is_editing():
            self.view.editor.redo()
            return
        history = self._history()
        label = history.redo(self) if history else None
        self.statusBar().showMessage(f"Redid: {label}" if label else "Nothing to redo")
        self._update_history_actions()

    # -- right-click menus ----------------------------------------------

    def _box_menu(self, menu: QMenu, runtime) -> None:
        """The things you can do to one piece of lettering."""
        edit = menu.addAction("Edit text")
        edit.triggered.connect(lambda: self.view.start_edit(runtime))

        # The shortcut is written into the label rather than bound to the
        # action: the canvas already handles the key, and only when you are not
        # typing. Qt right-aligns whatever follows a tab.
        read = menu.addAction("Copy original text\tCtrl+Shift+C")
        read.setToolTip(
            "Read the original lettering in this balloon and put it on the "
            "clipboard, for pasting into a dictionary"
        )
        read.triggered.connect(lambda: self._copy_original_text(runtime))

        menu.addSeparator()
        remove = menu.addAction("Delete box\tDel")
        remove.triggered.connect(
            lambda: (self.inspector.show_box(runtime), self._delete_box())
        )

    def canvas_menu(self, runtime) -> QMenu:
        """Build the right-click menu. Kept apart from showing it, because a
        menu that has been exec'd cannot be inspected -- by a test or anyone."""
        menu = QMenu(self)
        if runtime is not None:
            self.view.commit_edit()
            self.lettering.selected_id = runtime.box.id
            self.lettering.update()
            # The ordinary selection path, so the inspector, the list and the
            # undo grouping all react exactly as they do to a left click.
            self.view.selection_changed.emit(runtime)
            self._box_menu(menu, runtime)
            menu.addSeparator()

        area = menu.addAction("Copy text from an area…")
        area.setCheckable(True)
        area.setChecked(self.lookup_action.isChecked())
        area.setToolTip(
            "Drag a rectangle over any text to read it onto the clipboard"
        )
        area.toggled.connect(self.lookup_action.setChecked)

        # Which language the *original* is in. Nothing about reading a page is
        # Japanese by nature; the tag used to be written into the code, which
        # quietly shut out anyone translating anything else.
        languages = ocr.available_languages()
        if len(languages) > 1:
            current = ocr.language()
            picker = menu.addMenu("Read text in")
            for tag, name in languages:
                entry = picker.addAction(name)
                entry.setCheckable(True)
                entry.setChecked(tag == current)
                entry.triggered.connect(
                    lambda _checked=False, chosen=tag: self._set_ocr_language(chosen)
                )

        hide = menu.addAction("Hide lettering\tH")
        hide.setCheckable(True)
        hide.setChecked(self.hide_action.isChecked())
        hide.toggled.connect(self.hide_action.setChecked)
        return menu

    def _set_ocr_language(self, tag: str) -> None:
        prefs.set_ocr_language(tag)
        self.statusBar().showMessage(f"Reading pages as {ocr.language_name(tag)}")

    def _show_canvas_menu(self, position, runtime) -> None:
        self.canvas_menu(runtime).exec(position)

    def list_menu(self, position) -> QMenu | None:
        item = self.boxes_list.itemAt(position)
        runtime = self._runtime_from_row(item)
        if runtime is None:
            return None
        self._box_row_chosen(item)
        menu = QMenu(self)
        self._box_menu(menu, runtime)
        return menu

    def _show_list_menu(self, position) -> None:
        menu = self.list_menu(position)
        if menu is not None:
            menu.exec(self.boxes_list.viewport().mapToGlobal(position))

    # -- reading the original lettering ---------------------------------

    def _toggle_lookup(self, active: bool) -> None:
        self.view.lookup_active = bool(active)
        if active and self.view.stamp_active:
            self.stamp_action.setChecked(False)  # one drag tool at a time
        self.view.setCursor(Qt.CrossCursor if active else Qt.ArrowCursor)
        if not active:
            self.view.unsetCursor()
        self.statusBar().showMessage(
            "Drag over any text to copy it" if active else ""
        )

    def _offer_ocr_install(self) -> None:
        """Explain why nothing can be read, and offer to fix it in one press.

        The fix is a five-package pip line that has to name the right
        interpreter. That is a lot to retype correctly, and getting it subtly
        wrong installs into a Python the program is not running under -- which
        looks exactly like not having installed it at all.
        """
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle("Copy text")
        box.setText("No text recognition is available here.")
        box.setInformativeText(
            "I can install the Windows bindings now — about 5 MB, and they use "
            "the language packs already on this machine.\n\n"
            "It installs into the Python this program is running under, which "
            "is the part that is easy to get wrong by hand."
        )
        box.setDetailedText(
            ocr.install_hint()
            + "\n\nThe command that will run:\n  "
            + " ".join(ocr.pip_command())
        )
        install = box.addButton("Install now", QMessageBox.AcceptRole)
        box.addButton("Not now", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is not install:
            return

        self.statusBar().showMessage("Installing the OCR bindings…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            ok, detail = ocr.install_windows_ocr()
        finally:
            QApplication.restoreOverrideCursor()

        if ok:
            self.statusBar().showMessage("Installed — restart to use text copying")
            QMessageBox.information(
                self,
                "Copy text",
                f"Installed ({detail}).\n\n"
                "Please restart the program before using it.\n\n"
                "Python loaded — and failed to find — these libraries when it "
                "started, and a session that has had them installed underneath "
                "it is not the same as one that started with them. Reading can "
                "hang in that state.",
            )
        else:
            self.statusBar().showMessage("Could not install the OCR bindings")
            QMessageBox.warning(
                self, "Copy text", "That did not work.\n\n" + detail
            )

    def _copy_original_text(self, what) -> None:
        """Read a balloon or a dragged rectangle, and put the words on the clipboard."""
        if self.colour is None:
            return
        if not ocr.available():
            self._offer_ocr_install()
            return

        engine = ocr.engine_name()
        self.statusBar().showMessage(f"Reading with {engine}…")
        QApplication.processEvents()

        where = "that area" if isinstance(what, tuple) else "the balloon"
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if isinstance(what, tuple):
                text = ocr.read(self.colour, rect=what)
            else:
                # The balloon's own shape, not its bounding box: neighbouring
                # balloons overlap that rectangle and their words would come
                # back mixed in with the ones asked for.
                text = ocr.read(self.colour, mask=what.mask)
        except ocr.OcrTimeout as expired:
            self.statusBar().showMessage("The OCR engine did not answer")
            QMessageBox.warning(self, "Copy text", str(expired))
            return
        except Exception as error:  # noqa: BLE001 - shown rather than swallowed
            self.statusBar().showMessage("Could not read that")
            QMessageBox.warning(
                self, "Copy text", f"{type(error).__name__}: {error}"
            )
            return
        finally:
            QApplication.restoreOverrideCursor()

        if not text:
            self.statusBar().showMessage(f"Nothing readable in {where}")
            return

        QApplication.clipboard().setText(text)
        shown = text if len(text) <= 60 else text[:57] + "…"
        self.statusBar().showMessage(f"Copied: {shown}")

    def _lettering_hidden(self, hidden: bool) -> None:
        self.hide_action.blockSignals(True)
        self.hide_action.setChecked(hidden)
        self.hide_action.blockSignals(False)
        self.statusBar().showMessage(
            "Lettering hidden — press H to bring it back"
            if hidden
            else "Lettering shown"
        )

    def _update_history_actions(self) -> None:
        history = self._history()
        self.undo_action.setEnabled(bool(history and history.can_undo))
        self.redo_action.setEnabled(bool(history and history.can_redo))

    # -- fonts ----------------------------------------------------------

    def _load_font_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load font", str(self.project.folder), "Fonts (*.ttf *.otf *.ttc)"
        )
        if not path:
            return
        families = self._register_font_file(path)
        if not families:
            QMessageBox.warning(self, "Font", "That font could not be loaded.")
            return

        self._refresh_font_list()
        if families:
            self.inspector.font_box.setCurrentText(families[0])
            self.statusBar().showMessage(
                f"Loaded {families[0]} — press “Letter in this by default” to use it "
                "by default."
            )

    def _use_font_everywhere(self) -> None:
        """Adopt the selected family for new boxes and as the fallback."""
        family = self.inspector.font_box.currentText()
        if not family:
            return

        prefs.set_font_family(family)
        # Remember the file it came from, not just the name. Qt loses
        # application fonts when the process ends, and a remembered name with
        # nothing behind it is worse than useless -- it falls back in silence.
        # This matters most for a font picked up from a project folder: without
        # it, opening a different folder next session would lose the font.
        source = self._font_sources.get(family)
        if source:
            prefs.remember_font_file(source)
        set_chosen_font(family)
        self._show_chosen_font()

        # Boxes already on the page that asked for a font this machine does not
        # have were falling back; they resolve to the new choice now, so their
        # layout has to be redone.
        if self.view.lettering is not None:
            for runtime in self.view.lettering.runtimes:
                runtime.relayout(force=True)
            self.view.lettering.update()

        self.statusBar().showMessage(f"New balloons will be lettered in {family}")

    # -- output ---------------------------------------------------------

    def export_scale(self) -> int:
        return int(self.scale_box.currentData() or 1)

    # Beyond about this many pixels an export needs the better part of a
    # gigabyte for the canvas alone, which is worth a word before it is
    # attempted rather than an error afterwards.
    LARGE_EXPORT_PIXELS = 80_000_000

    def _confirm_export_size(self, scale: int) -> bool:
        """Warn before an export large enough to be a problem."""
        if scale == 1 or self.colour is None:
            return True
        height, width = self.colour.shape[:2]
        pixels = width * height * scale * scale
        if pixels <= self.LARGE_EXPORT_PIXELS:
            return True

        answer = QMessageBox.question(
            self,
            "Export",
            f"{scale}x of this page is {width * scale} × {height * scale} "
            f"— about {pixels / 1_000_000:.0f} megapixels.\n\n"
            "That will take a while and a good deal of memory. Carry on?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def render_page(
        self,
        path: Path,
        boxes: list[TextBox],
        gray: np.ndarray,
        layer=None,
        colour: np.ndarray | None = None,
        scale: int = 1,
    ) -> QImage:
        """Composite lettering onto a page, optionally at 2x or 4x.

        Upscaling is not the same job for the two halves of the picture. The
        artwork can only be interpolated -- there is no more of it than there
        was. The lettering is outlines, so at 2x it is drawn at 2x and is
        genuinely twice as sharp rather than twice as big. That is why this
        scales the painter instead of enlarging a finished export.
        """
        if colour is None:
            colour = load_colour(str(path))

        height, width = colour.shape[:2]
        # The enlarged copy is the canvas and *only* the canvas. Everything that
        # measures the page -- the masks, the colour a flat erase takes, the
        # tone a rebuild samples -- works against `gray`, which is not enlarged,
        # so handing any of them the big one lines nothing up. Overwriting
        # `colour` here once cost an export a crash and, worse, silently sampled
        # its erase colours from whatever happened to sit at twice the offset.
        if scale != 1:
            # Lanczos over Qt's own smooth scaling: on line art it holds an
            # edge together where bilinear turns it to porridge.
            enlarged = cv2.resize(
                colour,
                (width * scale, height * scale),
                interpolation=cv2.INTER_LANCZOS4,
            )
        else:
            enlarged = colour

        # Held on to for as long as the QImage that wraps it.
        buffer = np.ascontiguousarray(enlarged)
        rows, columns = buffer.shape[:2]
        base = QImage(buffer.data, columns, rows, 3 * columns, QImage.Format_BGR888)
        canvas = base.convertToFormat(QImage.Format_RGB32)

        painter = QPainter(canvas)
        painter.setRenderHints(
            QPainter.Antialiasing
            | QPainter.TextAntialiasing
            | QPainter.SmoothPixmapTransform
        )
        # Everything below works in page coordinates and lands on the bigger
        # canvas, so nothing else has to know the export was enlarged.
        if scale != 1:
            painter.scale(scale, scale)

        # Same three passes as the screen, so what you export is what you saw.
        runtimes = []
        for box in boxes:
            runtime = BoxRuntime(box, gray, colour)
            runtime.ensure_repair()
            runtimes.append(runtime)

        for runtime in runtimes:
            runtime.paint_erase(painter)

        if layer is not None and layer.dirty is not None:
            x0, y0, x1, y1 = layer.dirty
            painter.drawImage(
                QRectF(x0, y0, x1 - x0, y1 - y0), layer.region_image(x0, y0, x1, y1)
            )

        for runtime in runtimes:
            runtime.paint_text(painter)
        painter.end()
        return canvas

    def _export_page(self) -> None:
        if self.page_name is None:
            return
        self.view.commit_edit()

        source = self.project.folder / self.page_name
        suggested = str(
            self.project.folder / f"{Path(self.page_name).stem}{EXPORT_SUFFIX}.png"
        )
        target, _ = QFileDialog.getSaveFileName(self, "Export page", suggested, "PNG (*.png)")
        if not target:
            return

        scale = self.export_scale()
        if not self._confirm_export_size(scale):
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            image = self.render_page(
                source,
                self.project.page(self.page_name).boxes,
                self.gray,
                self.lettering.repair if self.lettering else None,
                self.colour,
                scale,
            )
        except Exception as error:  # noqa: BLE001 - shown rather than swallowed
            QMessageBox.warning(
                self, "Export",
                f"That did not work at {scale}x.\n\n"
                f"{type(error).__name__}: {error}\n\n"
                "A smaller size may fit where this one does not.",
            )
            return
        finally:
            QApplication.restoreOverrideCursor()
        image.save(target)
        self.statusBar().showMessage(
            f"Exported {target}" + (f" at {scale}x" if scale != 1 else "")
        )

    def _export_all(self) -> None:
        self.view.commit_edit()
        folder = QFileDialog.getExistingDirectory(self, "Export all pages to…")
        if not folder:
            return

        scale = self.export_scale()
        count = 0
        for path in self.project.image_files():
            boxes = self.project.page(path.name).boxes
            # Colour here too: a page exported without being opened must come
            # out the same as one exported from the canvas.
            colour = load_colour(str(path))
            gray = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
            layer = self._load_repair(path.name, colour)
            if not boxes and layer.is_empty:
                continue
            image = self.render_page(path, boxes, gray, layer, colour, scale)
            image.save(str(Path(folder) / f"{path.stem}{EXPORT_SUFFIX}.png"))
            count += 1
        self.statusBar().showMessage(
            f"Exported {count} page(s) to {folder}"
            + (f" at {scale}x" if scale != 1 else "")
        )

    # -- persistence ----------------------------------------------------

    def _save_quietly(self) -> None:
        self.view.commit_edit()
        try:
            self.project.save()
            self._save_repairs()
        except OSError:
            pass

    def _save(self) -> None:
        self.view.commit_edit()
        self.project.save()
        self._save_repairs()
        self.statusBar().showMessage(f"Saved {self.project.path.name}")

    def closeEvent(self, event) -> None:
        self._save_quietly()
        super().closeEvent(event)
