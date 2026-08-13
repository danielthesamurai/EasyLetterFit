"""Project state: text boxes and their persistence.

Source images are never modified. A project is a JSON sidecar next to the pages
plus, once repairs land, a per-page overlay image. Balloon masks are not stored
-- they are re-derived from the click seed, which is deterministic because
detection always runs against the untouched original pixels.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

PROJECT_FILENAME = "comic_translation.json"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# Exports are written beside the pages, so they must not be picked up as pages
# themselves on the next run.
EXPORT_SUFFIX = "_translated"

IGNORED_FOLDERS = {"repairs", "__pycache__", "app"}

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)"

# No particular font is required, and none is assumed. Until one is chosen these
# are tried in turn -- faces that ship with common systems and do not look
# absurd on comic lettering. Whichever turns up is a stand-in, and the program
# says so rather than pretending it is the right answer.
DEFAULT_FONT_CANDIDATES = (
    "Comic Sans MS",
    "Segoe Print",
    "Bradley Hand ITC",
    "DejaVu Sans",
    "Verdana",
)


def natural_key(name: str) -> tuple:
    """Sort key that reads runs of digits as numbers.

    Plain text order puts page 10 between 1 and 2, which is wrong for every
    comic ever numbered. Splitting on digit runs alternates text, number, text,
    so the same position in two keys always holds the same type and the tuples
    compare safely.

    The raw name goes last to break ties, so "01" and "1" keep a stable order
    rather than depending on how the directory happened to be read.
    """
    parts = re.split(r"(\d+)", name)
    return (
        tuple(int(part) if index % 2 else part.lower() for index, part in enumerate(parts)),
        name,
    )


@dataclass
class TextBox:
    """One editable piece of lettering on a page."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    text: str = ""
    kind: str = "bubble"  # "bubble" -> mask from seed; "free" -> plain rectangle

    seed: list[int] | None = None  # click point that found the balloon
    rect: list[float] | None = None  # x, y, w, h for free boxes

    # Empty means "whatever the lettering font is", resolved when drawn. A box
    # only records a family of its own once one is deliberately chosen for it.
    font_family: str = ""
    font_size: float = 36.0
    auto_fit: bool = True
    bold: bool = False
    italic: bool = False
    line_spacing: float = 1.0
    # Most lines the text may wrap onto; 0 leaves it to the shape. Capping it
    # asks for width rather than height -- see fit_font_size.
    max_lines: int = 0
    align: str = "center"
    angle: float = 0.0  # degrees, clockwise, about the region's centre
    vertical: bool = False  # tategaki: columns top-to-bottom, right-to-left

    color: str = "#000000"
    outline_color: str = "#FFFFFF"
    outline_width: float = 0.0

    padding: float = 14.0
    # Pixels to widen / heighten the area text flows into, beyond the shape the
    # artwork gives. Negative pulls it in. Affects layout only -- the erase
    # still follows the balloon, so stretching cannot paint outside it.
    stretch_x: float = 0.0
    stretch_y: float = 0.0
    erase: bool = True  # clear the original lettering underneath
    # Pixels to move each edge of the *erased* area outward (positive) or
    # inward (negative). Separate from stretch_x/stretch_y, which move the area
    # the text flows into: covering the original and placing the translation are
    # different jobs and do not always want the same rectangle. These may take
    # the erase outside the balloon, which is the point -- original lettering
    # does not always sit inside what detection found.
    erase_left: float = 0.0
    erase_right: float = 0.0
    erase_top: float = 0.0
    erase_bottom: float = 0.0
    # "white" fills the region -- right inside a balloon, where the paper is
    # blank. "rebuild" reconstructs the background, for text sitting on artwork.
    erase_mode: str = "white"
    offset: list[float] = field(default_factory=lambda: [0.0, 0.0])

    # Set the first time the text is actually changed. Selecting a balloon must
    # not disturb the page, so erasing waits for a deliberate edit.
    touched: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TextBox":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class PageState:
    boxes: list[TextBox] = field(default_factory=list)


class Project:
    """All pages in a folder and the lettering placed on them."""

    def __init__(self, folder: Path):
        # Resolved, so browsing to the parent works and the window title has a
        # name to show even when the app was started with a relative path.
        try:
            self.folder = Path(folder).resolve()
        except OSError:
            self.folder = Path(folder)
        self.pages: dict[str, PageState] = {}

    @property
    def path(self) -> Path:
        return self.folder / PROJECT_FILENAME

    def page(self, name: str) -> PageState:
        return self.pages.setdefault(name, PageState())

    def image_files(self) -> list[Path]:
        return sorted(
            (
                p
                for p in self.folder.iterdir()
                if p.is_file()
                and p.suffix.lower() in IMAGE_EXTENSIONS
                and not p.stem.endswith(EXPORT_SUFFIX)
            ),
            key=lambda p: natural_key(p.name),
        )

    def subfolders(self) -> list[Path]:
        """Child folders that could hold their own project."""
        try:
            children = list(self.folder.iterdir())
        except OSError:
            return []
        return sorted(
            (
                p
                for p in children
                if p.is_dir()
                and p.name not in IGNORED_FOLDERS
                and not p.name.startswith(".")
            ),
            key=lambda p: natural_key(p.name),
        )

    def save(self) -> None:
        # A box that was never typed into is not work, it is a click that went
        # nowhere. Saving those accumulates invisible boxes on top of the
        # balloons that produced them, and a later click then selects one of
        # them instead of doing anything -- which looks exactly like the tool
        # ignoring you.
        pages = {}
        for name, state in self.pages.items():
            kept = [b.to_dict() for b in state.boxes if b.text.strip() or b.touched]
            if kept:
                pages[name] = {"boxes": kept}
        # Browsing through folders should not leave a trail of empty project
        # files behind. Only write once there is work to record, or to update a
        # file that already exists.
        if not pages and not self.path.exists():
            return

        payload = {"version": 1, "pages": pages}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path) -> "Project":
        project = cls(folder)
        path = project.path
        if not path.exists():
            return project

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return project

        for name, data in payload.get("pages", {}).items():
            state = PageState()
            state.boxes = [TextBox.from_dict(b) for b in data.get("boxes", [])]
            project.pages[name] = state
        return project
