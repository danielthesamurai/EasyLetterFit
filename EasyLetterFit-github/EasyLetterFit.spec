# PyInstaller build for the downloadable release.
#
# One folder rather than one file, for two reasons. Qt is LGPL, which asks that
# anyone be able to replace it with their own build -- loose DLLs beside the
# program satisfy that where a self-extracting single file muddies it. And a
# folder starts immediately, where a single file unpacks itself to a temporary
# directory on every launch.
#
#     pyinstaller EasyLetterFit.spec
#
# Result: dist/EasyLetterFit/EasyLetterFit.exe

# Qt ships a great deal that a lettering tool never touches. Naming them here
# keeps the download to something a person will actually wait for.
UNUSED_QT = [
    "PySide6.Qt3DAnimation", "PySide6.Qt3DCore", "PySide6.Qt3DExtras",
    "PySide6.Qt3DInput", "PySide6.Qt3DLogic", "PySide6.Qt3DRender",
    "PySide6.QtBluetooth", "PySide6.QtCharts", "PySide6.QtDataVisualization",
    "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets", "PySide6.QtNetworkAuth", "PySide6.QtNfc",
    "PySide6.QtOpenGL", "PySide6.QtOpenGLWidgets", "PySide6.QtPdf",
    "PySide6.QtPdfWidgets", "PySide6.QtPositioning", "PySide6.QtQml",
    "PySide6.QtQuick", "PySide6.QtQuick3D", "PySide6.QtQuickControls2",
    "PySide6.QtQuickWidgets", "PySide6.QtRemoteObjects", "PySide6.QtScxml",
    "PySide6.QtSensors", "PySide6.QtSerialPort", "PySide6.QtSpatialAudio",
    "PySide6.QtSql", "PySide6.QtStateMachine", "PySide6.QtTest",
    "PySide6.QtTextToSpeech", "PySide6.QtWebChannel", "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineQuick", "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebSockets", "PySide6.QtCharts", "PySide6.QtUiTools",
]

# Not imported by anything here, and each drags in a great deal.
UNUSED_OTHER = ["tkinter", "matplotlib", "scipy", "pandas", "IPython", "pytest"]

# Qt's plugins, gathered by hand.
#
# PyInstaller's PySide6 hook did not collect them from this version, and the
# result is not a missing-plugin message: Qt cannot create a window at all
# without a platform plugin, and the process dies with a stack-buffer-overrun
# fail-fast before Python prints anything. Worth collecting explicitly rather
# than trusting a hook to keep pace with PySide6's releases.
#
#   platforms    -- qwindows.dll; without it there is no window at all
#   imageformats -- JPEG above all; half of comics are .jpg
#   styles       -- so the controls look like the rest of the system
#   iconengines  -- icons in the toolbar and dialogs
#
# Two are named individually because they are not free. qpdf drags in the whole
# 4.4 MB PDF engine to open a format no comic arrives in, and qdirect2d is a
# second Windows backend nobody selects when qwindows is present. qoffscreen
# stays: --self-test runs on it.
UNWANTED_PLUGINS = {"qpdf.dll", "qdirect2d.dll", "qicns.dll", "qwbmp.dll"}

import pathlib

import PySide6

_PLUGIN_ROOT = pathlib.Path(PySide6.__file__).parent / "plugins"
QT_PLUGINS = [
    (str(dll), f"PySide6/plugins/{group}")
    for group in ("platforms", "imageformats", "styles", "iconengines")
    for dll in (_PLUGIN_ROOT / group).glob("*.dll")
    if dll.name not in UNWANTED_PLUGINS
]
if not any(dest.endswith("platforms") for _, dest in QT_PLUGINS):
    raise SystemExit(
        "No Qt platform plugin found under "
        f"{_PLUGIN_ROOT} -- the build would produce a program that cannot open "
        "a window. Check the PySide6 layout before shipping."
    )

analysis = Analysis(
    ["run.py"],
    pathex=[],
    binaries=QT_PLUGINS,
    # The licence travels with the program, as the GPL asks.
    datas=[
        ("LICENSE", "."),
        ("THIRD-PARTY.md", "."),
        # The window and taskbar icon. The one baked into the .exe below covers
        # the file in Explorer; this covers the running program.
        ("EasyLetterFit.ico", "."),
    ],
    # The WinRT bindings are found only at runtime, inside functions, so
    # PyInstaller cannot see them by reading the source.
    hiddenimports=[
        "winrt.windows.foundation",
        "winrt.windows.foundation.collections",
        "winrt.windows.globalization",
        "winrt.windows.graphics.imaging",
        "winrt.windows.media.ocr",
        "winrt.windows.storage.streams",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=UNUSED_QT + UNUSED_OTHER,
    noarchive=False,
    optimize=0,
)

# Large files pulled in by dependencies that this program never reaches for.
#
#   opencv_videoio_ffmpeg -- OpenCV's video reader; no video is ever opened
#   _avif                 -- a Pillow codec, and Pillow is here only to hand an
#                            image to manga-ocr
#   opengl32sw            -- 20 MB of software OpenGL. Qt keeps it so that Qt
#                            Quick can draw on a machine with no usable GPU
#                            driver. Nothing here draws in OpenGL; the widgets
#                            are painted by Qt's raster engine.
#   Qt6Pdf                -- the PDF engine, reached only through the qpdf image
#                            plugin dropped above
#
# What is *not* trimmed: cv2.pyd, at 82 MB by far the largest thing here. The
# headless build of OpenCV is 81.9 MB -- measured, not assumed -- so swapping to
# it would buy nothing. Its size is the floor for this program.
UNUSED_BINARIES = (
    "opencv_videoio_ffmpeg", "_avif", "opengl32sw", "Qt6Pdf",
    # These two came from C:\Program Files\Git\mingw64\bin, picked up off PATH
    # rather than from anything this program depends on. Python brings its own
    # OpenSSL (libcrypto-3.dll, no -x64), which is what _ssl and _hashlib
    # actually load. Shipping a second copy of somebody else's build is 6.5 MB
    # of nothing, and it makes the release depend on what happens to be
    # installed on the machine that built it.
    "libcrypto-3-x64", "libssl-3-x64",
)
analysis.binaries = TOC(
    entry for entry in analysis.binaries
    if not any(part in entry[0] for part in UNUSED_BINARIES)
)

pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="EasyLetterFit",
    icon="EasyLetterFit.ico",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # a lettering tool has nothing to say to a terminal
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="EasyLetterFit",
)
