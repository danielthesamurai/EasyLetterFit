"""Reading the original lettering off the page, for looking a word up.

This is the one place the program reads text rather than drawing it, and it
exists for a narrow purpose: you meet a word you do not know, and you want it in
a dictionary without typing it out stroke by stroke. What it reads goes to the
clipboard, never into a text box -- the translation is still yours to write.

Two engines, whichever is present:

* **manga-ocr**, if it is installed and the language is Japanese. Trained on
  real manga, so it handles vertical text, furigana and stylised lettering far
  better than anything general-purpose. Large, and slow to load the first time.
* **Windows OCR**, which needs a language pack for whatever you are reading and
  a few megabytes of bindings. General-purpose, so it is weaker on comic
  lettering, but it is already on the machine and answers in milliseconds.

manga-ocr wins when both are there *and* the language is Japanese, which is the
only one it knows. Which language to read is a setting, not a fact about the
program. Neither is a dependency: without them the
feature reports that it cannot read anything, and the rest of the program is
unaffected.
"""

from __future__ import annotations

import asyncio
import threading

import cv2
import numpy as np

# Below this, characters are too few pixels across for either engine to be sure
# of them. Comic lettering on a web-sized page is routinely 12 px tall.
TARGET_SHORT_SIDE = 320
MAX_UPSCALE = 4.0

# And an upper limit. Reading is for a balloon or a phrase, but nothing stops a
# drag covering half a page, and a general-purpose OCR engine handed a few
# megapixels can take a very long time about it. Above this the crop is scaled
# down, which loses nothing: the text was already far bigger than it needs to be.
MAX_LONG_SIDE = 2200

# Where the reader got to, for saying so when it never comes back. A hang that
# names the step it hung on is a bug report; one that does not is a guess.
_stage = "idle"

# Whether a real read has been proved to work. Remembered so the proof costs one
# test image per session rather than one per copy.
_verified: bool | None = None
_verified_why = ""

_manga_reader = None
_manga_checked = False


def _cjk(ch: str) -> bool:
    """Is this a character that is never spaced from its neighbours?"""
    if not ch or ch.isspace():
        return False
    code = ord(ch)
    return (
        0x3000 <= code <= 0x30FF      # punctuation, hiragana, katakana
        or 0x3400 <= code <= 0x4DBF   # kanji extension A
        or 0x4E00 <= code <= 0x9FFF   # kanji
        or 0xF900 <= code <= 0xFAFF   # compatibility ideographs
        or 0xFF00 <= code <= 0xFFEF   # full-width forms
        or 0xAC00 <= code <= 0xD7AF   # hangul
    )


def tidy(text: str) -> str:
    """Put the pieces back together the way the writing system spaces them.

    OCR reports lines, and inside a line it separates what it thinks are words.
    Japanese does not space its words, so those separators are noise -- but the
    same text may hold a Latin phrase that does need them. Space is dropped only
    where both sides are characters that never take one.
    """
    out: list[str] = []
    for piece in text.replace("\r", "\n").split("\n"):
        piece = piece.strip()
        if not piece:
            continue
        out.append(piece)

    joined = ""
    for piece in out:
        if joined and not (_cjk(joined[-1]) and _cjk(piece[0])):
            joined += " "
        joined += piece

    # And the same rule for the spaces inside a line.
    result = []
    for index, ch in enumerate(joined):
        if ch == " " and 0 < index < len(joined) - 1:
            if _cjk(joined[index - 1]) and _cjk(joined[index + 1]):
                continue
        result.append(ch)
    return "".join(result).strip()


# -- what is installed --------------------------------------------------


def _manga_ocr():
    """The manga-ocr reader, loaded once, or None if it is not installed."""
    global _manga_reader, _manga_checked
    if _manga_checked:
        return _manga_reader
    _manga_checked = True
    try:
        from manga_ocr import MangaOcr

        _manga_reader = MangaOcr()
    except Exception:
        _manga_reader = None
    return _manga_reader


def available_languages() -> list[tuple[str, str]]:
    """Every language this machine can read, as (tag, name)."""
    try:
        from winrt.windows.media.ocr import OcrEngine

        return [
            (language.language_tag, language.display_name)
            for language in OcrEngine.available_recognizer_languages
        ]
    except Exception:
        return []


def language() -> str:
    """The language tag to read in.

    Nothing here is Japanese by nature. The program was written against
    Japanese comics and had the tag written into it, which quietly shut out
    everyone translating anything else. A stored choice wins; otherwise take
    whatever the machine offers, preferring a language it is likely to be
    wanted for over the machine's own interface language.
    """
    from . import prefs

    supported = [tag for tag, _ in available_languages()]
    if not supported:
        return ""

    stored = prefs.ocr_language()
    if stored:
        for tag in supported:
            if tag.lower() == stored.lower() or tag.lower().startswith(
                stored.lower() + "-"
            ):
                return tag

    # Comics needing translation are overwhelmingly in these, and a machine that
    # has the pack installed at all probably installed it for that reason.
    for preferred in ("ja", "ko", "zh"):
        for tag in supported:
            if tag.lower().startswith(preferred):
                return tag
    return supported[0]


def language_name(tag: str) -> str:
    for candidate, name in available_languages():
        if candidate == tag:
            return name
    return tag or "none"


def _windows_available() -> bool:
    ok, detail = diagnose()
    return ok and detail.startswith("Windows OCR")


def diagnose() -> tuple[bool, str]:
    """Whether anything can read the chosen language, and if not, why not.

    Swallowing the reason was a mistake worth not repeating: "no OCR is
    available" reads the same whether nothing is installed, the wrong Python is
    running, or a real error was thrown, and the first thing anyone needs to
    know is which. The report names the interpreter, because a second Python
    without the bindings looks identical from the outside.
    """
    import importlib.util
    import sys

    notes = [f"Python in use: {sys.executable}"]

    try:
        spec = importlib.util.find_spec("manga_ocr")
    except Exception as error:
        spec = None
        notes.append(f"manga-ocr: could not be checked ({error})")
    else:
        if spec is None:
            notes.append("manga-ocr: not installed")
        elif not language().lower().startswith("ja"):
            notes.append(
                "manga-ocr: installed, but it reads Japanese only — "
                f"using Windows OCR for {language_name(language())}"
            )
        elif _manga_ocr() is not None:
            return True, "manga-ocr"
        else:
            notes.append("manga-ocr: installed, but the model failed to load")

    # Import every module the reader uses, not a representative couple. A set
    # missing one of them imports far enough to look installed and then fails
    # -- or worse, hangs -- at the moment it is finally needed.
    import importlib

    missing = []
    for name in WINDOWS_OCR_MODULES:
        try:
            importlib.import_module(name)
        except Exception as error:
            missing.append(f"{name} ({type(error).__name__}: {error})")

    if missing:
        notes.append("Windows OCR: these bindings are not importable here:")
        notes.extend(f"    {item}" for item in missing)
        notes.append("  fix with:\n    " + " ".join(pip_command()))
        return False, "\n".join(notes)

    try:
        from winrt.windows.globalization import Language
        from winrt.windows.media.ocr import OcrEngine
    except Exception as error:
        notes.append(
            f"Windows OCR: bindings not importable here "
            f"({type(error).__name__}: {error})\n"
            "  fix with:\n    " + " ".join(pip_command())
        )
        return False, "\n".join(notes)

    tag = language()
    try:
        if tag and OcrEngine.is_language_supported(Language(tag)):
            # Everything imports and the language is there. Neither of those
            # means a read will work, so do one -- once per session, and again
            # after anything is installed.
            global _verified, _verified_why
            if _verified is None:
                _verified, _verified_why = self_check()
            if _verified:
                return True, f"Windows OCR ({language_name(tag)})"
            notes.append(
                "Windows OCR: the bindings import and Japanese is present, but "
                "a test read failed:\n"
                f"    {_verified_why}\n"
                "  This usually means the set of winrt packages is incomplete. "
                "Uninstalling one\n  leaves its folder behind, so the import "
                "keeps working while the module is empty.\n"
                "  fix with:\n    " + " ".join(pip_command())
            )
            return False, "\n".join(notes)
        tags = [
            language.language_tag
            for language in OcrEngine.available_recognizer_languages
        ]
        notes.append(
            "Windows OCR: working, but Japanese is not one of its languages "
            f"({', '.join(tags) or 'none'}).\n"
            "  Settings → Time & language → Language → Japanese → Options, and "
            "add the optional OCR / handwriting component."
        )
    except Exception as error:
        notes.append(f"Windows OCR: failed ({type(error).__name__}: {error})")
    return False, "\n".join(notes)


def engine_name() -> str:
    """Which engine will be used, for saying so in the status bar."""
    ok, detail = diagnose()
    return detail if ok else ""


def available() -> bool:
    return diagnose()[0]


# The Windows bindings, split across packages the way the WinRT projection is.
# Nobody should have to type this list correctly by hand -- and getting it wrong
# is not obvious, because these packages do not declare each other. The OCR
# package requires only winrt-runtime, so pip will happily install a set that
# imports far enough to look healthy and then fails, or hangs, deeper in.
#
# Windows.Foundation is the one to lose sleep over: it carries the machinery an
# `await` on a WinRT operation runs on. Without it a call does not raise, it
# simply never completes.
WINDOWS_OCR_PACKAGES = (
    "winrt-runtime",
    "winrt-Windows.Foundation",
    "winrt-Windows.Foundation.Collections",
    "winrt-Windows.Media.Ocr",
    "winrt-Windows.Graphics.Imaging",
    "winrt-Windows.Storage.Streams",
    "winrt-Windows.Globalization",
)

# Every module the reader actually touches. Checking a couple of them and
# declaring the feature ready is what let a half-installed set look fine.
WINDOWS_OCR_MODULES = (
    "winrt.windows.foundation",
    "winrt.windows.globalization",
    "winrt.windows.graphics.imaging",
    "winrt.windows.media.ocr",
    "winrt.windows.storage.streams",
)


def pip_command() -> list[str]:
    """How to install the bindings *for the Python that is actually running*.

    Which interpreter matters more than it looks: a machine with several will
    happily install into one and run the program from another, and the symptom
    is this feature claiming nothing is installed when something clearly is.
    """
    import os
    import sys

    executable = sys.executable
    # pip works under pythonw, but a console build gives it somewhere to talk.
    if executable.lower().endswith("pythonw.exe"):
        console = executable[: -len("pythonw.exe")] + "python.exe"
        if os.path.exists(console):
            executable = console
    return [executable, "-m", "pip", "install", *WINDOWS_OCR_PACKAGES]


def install_windows_ocr(timeout: float = 900.0) -> tuple[bool, str]:
    """Install the Windows bindings and report whether reading works afterwards."""
    import importlib
    import subprocess

    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        done = subprocess.run(
            pip_command(),
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=flags,
        )
    except Exception as error:
        return False, f"Could not run pip: {type(error).__name__}: {error}"

    importlib.invalidate_caches()
    # Anything just installed deserves a fresh verdict, not one cached before it
    # existed.
    global _verified, _verified_why
    _verified, _verified_why = None, ""

    ok, detail = diagnose()  # which now proves itself with a real read
    if ok:
        return True, detail

    tail = (done.stderr or done.stdout or "").strip().splitlines()
    return False, "\n".join([detail, "", *tail[-6:]])


def self_check() -> tuple[bool, str]:
    """Read a picture made here, to prove the whole path actually runs.

    This is the only check worth trusting, and the reason is unpleasant:
    uninstalling a winrt package leaves its directory behind, so
    `import winrt.windows.foundation` keeps succeeding against a module with
    nothing in it. Every import-based test therefore passes on a broken
    install, which is exactly how this feature came to report itself healthy
    and then hang.

    Goes straight to the engine rather than through read(), which would ask
    whether OCR is available and arrive back here.
    """
    canvas = np.full((160, 420, 3), 255, np.uint8)
    cv2.putText(canvas, "TEST", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 3.0,
                (0, 0, 0), 8)
    crop = crop_for_ocr(canvas, rect=(0, 0, 420, 160))
    if crop is None:
        return False, "could not prepare a test image"
    try:
        text = tidy(_windows_call(crop, timeout=min(READ_TIMEOUT, 15.0)))
    except OcrTimeout as expired:
        return False, str(expired).splitlines()[0]
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"
    if not text.strip():
        return False, "the engine ran but read nothing from a test image"
    return True, text.strip()


def install_hint() -> str:
    """Why nothing can be read, in terms that point at the fix."""
    ok, detail = diagnose()
    if ok:
        return f"Text recognition is available ({detail})."
    return (
        "No Japanese OCR is available here.\n\n"
        f"{detail}\n\n"
        "manga-ocr reads comic lettering considerably better than the Windows "
        "engine, if you would rather have that:\n"
        "    pip install manga-ocr"
    )


# -- preparing the picture ----------------------------------------------


def crop_for_ocr(
    page: np.ndarray, mask: np.ndarray | None = None, rect: tuple | None = None
) -> np.ndarray | None:
    """The pixels to read: just this region, on a blank field.

    Masking matters as much as cropping. A balloon's bounding box overlaps its
    neighbours, and handing the engine that rectangle gets the neighbour's words
    back mixed in with the ones asked for.
    """
    if mask is not None:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
    elif rect is not None:
        x, y, w, h = (int(round(v)) for v in rect)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(page.shape[1], x + w), min(page.shape[0], y + h)
    else:
        return None
    if x1 <= x0 or y1 <= y0:
        return None

    crop = page[y0:y1, x0:x1]
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    crop = crop.copy()

    if mask is not None:
        inside = mask[y0:y1, x0:x1] > 0
        # Paper white, so anything outside the region reads as blank rather than
        # as a shape the engine might try to make a letter of.
        crop[~inside] = (255, 255, 255)

    # A margin, because engines look for text with space around it.
    crop = cv2.copyMakeBorder(crop, 12, 12, 12, 12, cv2.BORDER_CONSTANT,
                              value=(255, 255, 255))

    short = min(crop.shape[:2])
    if short < TARGET_SHORT_SIDE:
        scale = min(MAX_UPSCALE, TARGET_SHORT_SIDE / max(1, short))
        crop = cv2.resize(
            crop,
            (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    longest = max(crop.shape[:2])
    if longest > MAX_LONG_SIDE:
        scale = MAX_LONG_SIDE / longest
        crop = cv2.resize(
            crop,
            (max(8, int(crop.shape[1] * scale)), max(8, int(crop.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return crop


# -- the engines --------------------------------------------------------


class OcrTimeout(RuntimeError):
    """The engine was asked and never answered."""


# Long enough for a slow first call, short enough that a wedged engine is an
# inconvenience rather than a lost afternoon.
READ_TIMEOUT = 20.0


def _windows_call(bgr: np.ndarray, timeout: float | None = None) -> str:
    """Run the WinRT read on a thread of its own, and never wait forever.

    Two reasons it does not run on the calling thread. The caller is the GUI
    thread inside Qt's event loop, which is already a COM apartment Qt set up
    for its own purposes, and blocking it on an asynchronous WinRT call is the
    kind of arrangement that works everywhere until it deadlocks on somebody
    else's machine -- as it did. A worker gets a plain multi-threaded apartment
    with nothing else in it.

    And whatever happens, this returns. A window that stops responding until it
    is killed is a far worse answer than "that took too long".

    The limit is read here rather than bound as a default argument, so that
    changing READ_TIMEOUT actually changes the timeout.
    """
    global _stage
    if timeout is None:
        timeout = READ_TIMEOUT
    _stage = "starting the reader thread"
    result: dict = {}

    def work() -> None:
        try:
            from winrt.runtime import ApartmentType, init_apartment

            init_apartment(ApartmentType.MULTI_THREADED)
        except Exception:
            pass  # already initialised, or an older binding without it
        try:
            result["text"] = asyncio.run(_windows_read(bgr))
        except BaseException as error:  # noqa: BLE001 - reported to the caller
            result["error"] = error
        else:
            try:
                from winrt.runtime import uninit_apartment

                uninit_apartment()
            except Exception:
                pass

    thread = threading.Thread(target=work, name="ocr-read", daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise OcrTimeout(
            f"The OCR engine did not answer within {timeout:.0f} seconds.\n\n"
            f"It was stuck at: {_stage}\n\n"
            "Please pass that line on — it names the step that hung, which is "
            "the one thing needed to fix this."
        )
    if "error" in result:
        raise result["error"]
    return result.get("text", "")


def _software_bitmap(bgr: np.ndarray):
    """Build a WinRT bitmap straight from the pixels, without touching a stream.

    The obvious way to hand a picture to Windows is to encode it as a PNG, push
    the bytes into an InMemoryRandomAccessStream and let a BitmapDecoder read
    them back. That is what this did, and on at least one machine it hung
    forever inside DataWriter.store_async -- a step whose only purpose was to
    undo an encoding this code had just done.

    Copying the pixels into a buffer is what the round-trip was for. It has no
    asynchronous calls in it at all, so there is nothing left to wait on, and it
    skips a PNG compress and decompress into the bargain.
    """
    from winrt.windows.graphics.imaging import (
        BitmapAlphaMode,
        BitmapPixelFormat,
        SoftwareBitmap,
    )
    from winrt.windows.storage.streams import Buffer

    bgra = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA))
    height, width = bgra.shape[:2]
    raw = bgra.tobytes()

    buffer = Buffer(len(raw))
    buffer.length = len(raw)
    memoryview(buffer)[:] = raw

    bitmap = SoftwareBitmap(
        BitmapPixelFormat.BGRA8, width, height, BitmapAlphaMode.PREMULTIPLIED
    )
    bitmap.copy_from_buffer(buffer)
    return bitmap


async def _windows_read(bgr: np.ndarray) -> str:
    global _stage

    _stage = "importing the bindings"
    from winrt.windows.globalization import Language
    from winrt.windows.media.ocr import OcrEngine

    _stage = f"handing the pixels to Windows ({bgr.shape[1]}x{bgr.shape[0]})"
    bitmap = _software_bitmap(bgr)

    _stage = "starting the recognition engine"
    tag = language()
    engine = OcrEngine.try_create_from_language(Language(tag)) if tag else None
    if engine is None:
        return ""

    _stage = "recognising"
    result = await engine.recognize_async(bitmap)
    _stage = "putting the characters in order"
    if not result.lines:
        return ""

    words = []
    for line in result.lines:
        for word in line.words or []:
            box = word.bounding_rect
            words.append((box.x, box.y, box.width, box.height, word.text))
    if not words:
        return "\n".join(line.text for line in result.lines)
    return _reading_order(words)


def _reading_order(words: list[tuple]) -> str:
    """Put the characters back in the order they are meant to be read.

    The engine groups what it finds into lines of its own choosing, and on
    vertical Japanese it sometimes groups *across* the columns instead of down
    them -- so a three-column balloon comes back as rows holding one character
    from each column, which is the right characters in an unreadable order.

    The word boxes are trustworthy even when the grouping is not, so the
    grouping is redone here from the geometry: gather characters into columns by
    where they sit, read each column downwards, and take the columns from the
    right. Text that is genuinely horizontal falls out of the same test, because
    its columns hold one or two characters rather than a sentence.
    """
    widths = sorted(w[2] for w in words)
    tolerance = max(4.0, widths[len(widths) // 2] * 0.6)

    by_x = sorted(words, key=lambda w: w[0] + w[2] / 2.0)
    columns: list[list[tuple]] = [[by_x[0]]]
    for word in by_x[1:]:
        centre = word[0] + word[2] / 2.0
        last = columns[-1][-1]
        if centre - (last[0] + last[2] / 2.0) > tolerance:
            columns.append([])
        columns[-1].append(word)

    stacked = sum(len(c) for c in columns) / len(columns)
    if len(columns) >= 2 and stacked >= 3:
        pieces = []
        for column in reversed(columns):  # right to left
            column.sort(key=lambda w: w[1])  # and downwards
            pieces.append("".join(w[4] for w in column))
        return "\n".join(pieces)

    # Ordinary rows: down the page, then across. Bucketing the tops keeps a line
    # together when its characters sit a pixel or two apart.
    heights = sorted(w[3] for w in words)
    row = max(1.0, heights[len(heights) // 2] * 0.6)
    ordered = sorted(words, key=lambda w: (round(w[1] / row), w[0]))

    pieces, current, last_row = [], [], None
    for word in ordered:
        here = round(word[1] / row)
        if last_row is not None and here != last_row:
            pieces.append(" ".join(current))
            current = []
        current.append(word[4])
        last_row = here
    if current:
        pieces.append(" ".join(current))
    return "\n".join(pieces)


def read(
    page: np.ndarray, mask: np.ndarray | None = None, rect: tuple | None = None
) -> str:
    """Read the lettering in one region. Returns "" when nothing was read."""
    crop = crop_for_ocr(page, mask, rect)
    if crop is None:
        return ""

    reader = None
    # Only for Japanese: manga-ocr is trained on it alone, and handed Korean or
    # Chinese it does not decline, it invents Japanese.
    if language().lower().startswith("ja"):
        try:
            import importlib.util

            if importlib.util.find_spec("manga_ocr") is not None:
                reader = _manga_ocr()
        except Exception:
            reader = None

    if reader is not None:
        from PIL import Image

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        try:
            return tidy(reader(Image.fromarray(rgb)))
        except Exception:
            return ""

    if not _windows_available():
        return ""
    # A timeout is the caller's business -- it means something is wrong rather
    # than that the balloon was empty, and it deserves saying out loud.
    return tidy(_windows_call(crop))
