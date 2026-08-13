"""Undo/redo for one page: text boxes and pixel repairs on a single stack.

Box changes are stored as snapshots. The state is a small list of plain
dataclasses -- a few hundred bytes per box -- so capturing all of it costs
nothing, and a snapshot cannot drift out of step with the real state the way
hand-written undo logic does.

Pixel repairs cannot work that way: a full copy of a 600 dpi repair layer is
~13 MB per stroke. Those store only the 64px tiles a stroke actually touched.

Both live on the same stack, because a user pressing Ctrl+Z means "the last
thing I did", not "the last thing of a particular kind".
"""

from __future__ import annotations

from typing import Callable, Protocol


class Context(Protocol):
    """What an entry needs from the editor to undo itself."""

    def capture_boxes(self) -> list: ...
    def restore_boxes(self, data: list) -> None: ...
    def swap_repair(self, tiles: dict) -> None: ...


class BoxesEntry:
    """A snapshot of every text box on the page."""

    def __init__(self, label: str, data: list):
        self.label = label
        self.data = data

    def revert(self, context: Context) -> "BoxesEntry":
        current = context.capture_boxes()
        context.restore_boxes(self.data)
        return BoxesEntry(self.label, current)


class RepairEntry:
    """The tiles one clone-stamp stroke overwrote."""

    def __init__(self, label: str, tiles: dict):
        self.label = label
        self.tiles = tiles

    def revert(self, context: Context) -> "RepairEntry":
        # Swapping leaves the entry holding what was just replaced, so the very
        # same object is what redo needs.
        context.swap_repair(self.tiles)
        return self


class History:
    """Undo and redo stacks for one page."""

    LIMIT = 100

    def __init__(self, capture: Callable[[], list]):
        self._capture = capture
        self._undo: list = []
        self._redo: list = []
        self._coalesce = None

    def snapshot(self, label: str, coalesce=None) -> None:
        """Record the boxes before changing them.

        `coalesce` collapses a run of related changes into one step: dragging a
        slider or holding an arrow key should be a single undo, not forty. The
        first call in a run captures the state; the rest are ignored until
        something with a different key happens.
        """
        if coalesce is not None and coalesce == self._coalesce:
            return
        self.push(BoxesEntry(label, self._capture()))
        self._coalesce = coalesce

    def push(self, entry) -> None:
        self._undo.append(entry)
        del self._undo[: -self.LIMIT]
        self._redo.clear()
        self._coalesce = None

    def break_coalescing(self) -> None:
        """End the current run, so the next change starts a fresh undo step."""
        self._coalesce = None

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self, context: Context) -> str | None:
        if not self._undo:
            return None
        entry = self._undo.pop()
        self._redo.append(entry.revert(context))
        self._coalesce = None
        return entry.label

    def redo(self, context: Context) -> str | None:
        if not self._redo:
            return None
        entry = self._redo.pop()
        self._undo.append(entry.revert(context))
        self._coalesce = None
        return entry.label
