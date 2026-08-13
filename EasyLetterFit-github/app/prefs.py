"""Preferences that outlive a single project.

Kept apart from Project: which font you letter in is a property of you, not of
the folder you happen to have open, and it should survive both switching
projects and restarting the program.

Stored as an ini file rather than the registry so it can be read, edited and
thrown away by hand.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings

ORGANISATION = "EasyLetterFit"
APPLICATION = "settings"

# What the settings were called before the program had a name. Read once, so an
# existing install keeps its font, language and last folder across the rename.
FORMER_ORGANISATION = "comic_translation"

FONT_EXTENSIONS = {".ttf", ".otf", ".ttc"}


def _settings() -> QSettings:
    current = QSettings(
        QSettings.IniFormat, QSettings.UserScope, ORGANISATION, APPLICATION
    )
    if not current.allKeys():
        _adopt_former(current)
    return current


def _adopt_former(current: QSettings) -> None:
    """Carry settings over from before the program was named.

    Renaming the settings key would otherwise silently reset everyone's chosen
    font, reading language and last folder -- a rename is not a reason to lose
    somebody's preferences.
    """
    former = QSettings(
        QSettings.IniFormat, QSettings.UserScope, FORMER_ORGANISATION, APPLICATION
    )
    keys = former.allKeys()
    if not keys:
        return
    for key in keys:
        current.setValue(key, former.value(key))
    current.sync()


def settings_path() -> str:
    return _settings().fileName()


def font_family() -> str:
    """The chosen lettering font, or "" if none has been picked."""
    return str(_settings().value("font/family", "") or "")


def set_font_family(family: str) -> None:
    _settings().setValue("font/family", family or "")


def font_files() -> list[str]:
    """Font files to register at startup, so a chosen font survives a restart.

    Qt forgets application fonts when the process ends, so remembering the
    family name alone would leave the next run falling back again.
    """
    stored = _settings().value("font/files", [])
    if isinstance(stored, str):  # a one-element list comes back as a bare string
        stored = [stored] if stored else []
    return [p for p in stored if Path(p).exists()]


def remember_font_file(path: str) -> None:
    files = font_files()
    if path not in files:
        files.append(path)
    _settings().setValue("font/files", files)


def forget_font_file(path: str) -> None:
    _settings().setValue("font/files", [p for p in font_files() if p != path])


def export_scale() -> int:
    """Size the last export was made at, so the choice sticks between sessions."""
    try:
        value = int(_settings().value("export/scale", 1))
    except (TypeError, ValueError):
        return 1
    return value if value in (1, 2, 4) else 1


def set_export_scale(scale: int) -> None:
    _settings().setValue("export/scale", int(scale))


def last_folder() -> str:
    """The folder last opened, so the program starts where you left off."""
    return str(_settings().value("session/folder", "") or "")


def set_last_folder(folder) -> None:
    _settings().setValue("session/folder", str(folder))


def ocr_language() -> str:
    """Language tag to read pages in, or "" to take whatever is available."""
    return str(_settings().value("ocr/language", "") or "")


def set_ocr_language(tag: str) -> None:
    _settings().setValue("ocr/language", tag or "")


def fonts_beside(folder: Path) -> list[Path]:
    """Font files sitting in a project folder.

    Dropping the .ttf next to the pages is the obvious thing to do, so treat it
    as an instruction rather than as clutter.
    """
    try:
        return sorted(
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in FONT_EXTENSIONS
        )
    except OSError:
        return []
