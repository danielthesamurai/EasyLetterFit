"""Launch EasyLetterFit, the comic translation letterer.

    python run.py [folder-of-pages]

Defaults to the folder this script lives in. Double-clicking
"EasyLetterFit.cmd" does the same thing without a console window.

    python run.py --self-test

builds the window without opening it and reports whether everything the program
needs is present. That is what to run when the launcher does nothing.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path


def has_console() -> bool:
    """Is there a window the user could read a printed message in?"""
    try:
        import ctypes

        return bool(ctypes.windll.kernel32.GetConsoleWindow())
    except Exception:
        return True  # not Windows: assume a terminal


def report(title: str, message: str) -> None:
    """Say something when there may be no console to say it to.

    Launched from a shortcut there is no window to print into, so a failure to
    start would otherwise look like nothing happening at all. Run from a
    terminal there is, and a modal box would only be in the way -- worse, it
    would hang anything that runs this without a person watching.
    """
    # A packaged windowed build has no standard streams at all -- sys.stderr is
    # None there -- and this is the one function that must not fail, since it is
    # how every other failure gets explained.
    try:
        sys.stderr.write(f"{title}\n\n{message}\n")
    except Exception:
        pass
    if has_console():
        return
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)
    except Exception:
        pass  # no user32 -- stderr will have to do


def beside_program(name: str) -> Path:
    """A file shipped with the program, wherever it happens to be living.

    Packaged, the datas land in PyInstaller's unpack folder rather than next to
    the .exe; from source they sit beside this file.
    """
    root = getattr(sys, "_MEIPASS", None) or Path(__file__).resolve().parent
    return Path(root) / name


def starting_folder() -> Path:
    """Where to look for pages when nowhere has been opened yet.

    Running from source, the folder this file sits in is a reasonable guess --
    that is where a person testing the program keeps a page or two. Packaged, it
    is not: `__file__` points inside the bundle's own `_internal`, so a first run
    would greet a new user with a list of stray icons from the libraries. The
    folder holding the program is empty of pages, which is the honest answer,
    and "Open folder…" is right there in the toolbar.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def main() -> int:
    try:
        from PySide6.QtWidgets import QApplication

        from app.mainwindow import MainWindow
    except Exception:
        report(
            "EasyLetterFit could not start",
            "Something it needs is missing or broken.\n\n"
            f"Python: {sys.executable}\n\n"
            f"{traceback.format_exc()}\n"
            "If PySide6 is not installed for this Python, install it with:\n"
            f'  "{sys.executable}" -m pip install PySide6 opencv-python numpy Pillow',
        )
        return 1

    arguments = [a for a in sys.argv[1:] if a != "--self-test"]
    self_test = "--self-test" in sys.argv
    if self_test:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

        # Settings go somewhere disposable for the duration. Opening a folder
        # records it as the last one opened, and a self-test has no business
        # moving where the program starts next time -- the first run of this
        # left the packaged build's own innards remembered as the working
        # folder, which is exactly the wrong thing for a diagnostic to do.
        from PySide6.QtCore import QSettings

        QSettings.setPath(
            QSettings.IniFormat,
            QSettings.UserScope,
            tempfile.mkdtemp(prefix="easyletterfit-selftest-"),
        )

    from app import prefs

    if arguments:
        folder = Path(arguments[0])
    else:
        # Where you were last, since a comic takes many sittings and the folder
        # the program happens to live in is nobody's working folder but the
        # author's. Falls back if that folder has since gone.
        remembered = prefs.last_folder()
        folder = (
            Path(remembered)
            if remembered and Path(remembered).is_dir()
            else starting_folder()
        )

    if not folder.is_dir():
        report("EasyLetterFit", f"Not a folder: {folder}")
        return 1

    app = QApplication(sys.argv)
    app.setApplicationName("EasyLetterFit")

    icon_file = beside_program("EasyLetterFit.ico")
    if icon_file.exists():
        from PySide6.QtGui import QIcon

        app.setWindowIcon(QIcon(str(icon_file)))

    try:
        window = MainWindow(folder)
    except Exception:
        report(
            "EasyLetterFit could not open that folder",
            f"{folder}\n\n{traceback.format_exc()}",
        )
        return 1

    if self_test:
        from app import ocr

        readable, detail = ocr.diagnose()
        lines = [
            f"ok: {sys.executable}",
            f"ok: window built, {window.pages.count()} rows listed for {folder}",
            f"{'ok' if readable else 'NO'}: text copying - "
            f"{detail if readable else 'unavailable'}",
        ]
        if not readable:
            lines += [f"    {line}" for line in detail.splitlines()]

        report_text = "\n".join(lines)
        try:
            print(report_text)
        except Exception:
            pass  # windowed build: no stdout to print to

        # A packaged build is windowed and has no console, so printing a
        # diagnosis into the void helps nobody -- and diagnosing a packaged
        # build is exactly when someone needs it. Put it somewhere they can
        # read, and say where.
        if not has_console():
            written = Path(tempfile.gettempdir()) / "easyletterfit-selftest.txt"
            try:
                written.write_text(report_text, encoding="utf-8")
            except OSError:
                written = None
            report(
                "EasyLetterFit self-test",
                report_text + (f"\n\nAlso written to:\n{written}" if written else ""),
            )
        return 0

    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
