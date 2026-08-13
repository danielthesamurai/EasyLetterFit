# Third-party components

EasyLetterFit itself is licensed under the GNU General Public License v3.0; see
`LICENSE`. The downloadable release also carries the libraries below.

| Component | Licence | Notes |
| --- | --- | --- |
| **Qt** and **PySide6** | LGPL v3 | See below. |
| **OpenCV** (`opencv-python`) | Apache License 2.0 | |
| **NumPy** | BSD 3-Clause | |
| **Pillow** | MIT-CMU | |
| **Python** | PSF License | |
| **winrt** bindings (optional) | MIT | Only present if text copying was set up. |
| **manga-ocr** (optional) | Apache License 2.0 | Only if installed separately. |

## Qt and the LGPL

Qt is used under the **LGPL v3**, which requires that you be able to replace it
with your own build of Qt.

The release is a **folder, not a single file**, precisely so you can: the Qt
DLLs sit beside the program as ordinary files, and replacing them with
compatible ones of your own is a matter of overwriting them. Nothing is
statically linked, packed, or hidden inside an archive that must be unpacked
first.

The Qt sources are available from <https://download.qt.io/>, and the LGPL v3
text from <https://www.gnu.org/licenses/lgpl-3.0.txt>.

GPL v3 — the licence of this program — is compatible with LGPL v3, so the
combination is distributable as a whole under GPL v3.

## Fonts

**No font is included.** The program letters in whatever face you choose, and
until you choose one it borrows something already installed on the machine and
says so. Any font you load stays yours; nothing is copied into the program or
its releases.
