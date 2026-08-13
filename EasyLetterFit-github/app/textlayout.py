"""Fit lines of text into an arbitrarily shaped region.

A speech balloon is an ellipse, a spiky shout, or a rounded blob -- never a
rectangle -- so wrapping to a bounding box wastes the middle and overflows the
top and bottom. This measures the width actually available at each height.

Lines are slices of the source string, never reassembled from words. That keeps
the spacing the author typed, and lets the caret sit at any offset including
inside a run of spaces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np
from PySide6.QtGui import QFont, QFontMetricsF


@dataclass
class Word:
    """A run of non-space text and where it came from in the source string."""

    text: str
    start: int
    end: int


@dataclass
class LaidGlyph:
    """One character placed on its own.

    Horizontal text is drawn a line at a time, but a vertical column stacks each
    character separately -- and some of them are rotated or nudged -- so vertical
    lines carry per-character placements instead.
    """

    text: str
    start: int  # source offset
    x: float
    baseline: float
    rotated: bool = False


@dataclass
class LaidLine:
    """One rendered line or column, tied back to the text it came from.

    `text` is what gets drawn. `raw` is the same slice plus whatever whitespace
    trails it, which is what caret positions are measured against -- so pressing
    space visibly moves the cursor even when it lands on a line break.
    """

    text: str
    raw: str
    start: int  # source offset of raw[0]
    x: float  # left edge, page coordinates
    baseline: float  # page coordinates
    glyphs: list = field(default_factory=list)  # set for vertical columns

    @property
    def end(self) -> int:
        return self.start + len(self.raw)


# Scripts that wrap between characters rather than at spaces. Japanese and
# Chinese are written without spaces at all, so word-based wrapping gives one
# enormous "word" that can never fit a balloon.
_PER_CHARACTER_RANGES = (
    (0x3000, 0x303F),  # CJK punctuation
    (0x3040, 0x309F),  # hiragana
    (0x30A0, 0x30FF),  # katakana
    (0x3400, 0x4DBF),  # CJK extension A
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xF900, 0xFAFF),  # compatibility ideographs
    (0xFF00, 0xFF9F),  # fullwidth forms and halfwidth katakana
    (0xAC00, 0xD7AF),  # hangul syllables
)

# Kinsoku shori: characters that may not begin a line (trailing punctuation,
# small kana, closing brackets) or end one (opening brackets).
NO_LINE_START = set(
    "、。，．,.!?:;)]}>»、。〉》」』】〕｝］）’”〟｠ゝゞヽヾーぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ"
)
NO_LINE_END = set("([{<«（［｛〈《「『【〔‘“〝｟")


def breaks_per_character(ch: str) -> bool:
    code = ord(ch)
    return any(low <= code <= high for low, high in _PER_CHARACTER_RANGES)


def _split_run(run: str, offset: int) -> list[Word]:
    """Break one run of non-space text into wrappable tokens.

    Latin stays whole words; CJK becomes one token per character, which is where
    Japanese and Chinese are allowed to wrap.
    """
    words: list[Word] = []
    buffer = ""
    buffer_start = 0

    for i, ch in enumerate(run):
        if breaks_per_character(ch):
            if buffer:
                words.append(Word(buffer, offset + buffer_start, offset + i))
                buffer = ""
            words.append(Word(ch, offset + i, offset + i + 1))
        else:
            if not buffer:
                buffer_start = i
            buffer += ch

    if buffer:
        words.append(Word(buffer, offset + buffer_start, offset + len(run)))
    return words


def tokenise(text: str) -> list[tuple[int, list[Word]]]:
    """Split into paragraphs, each as (source offset, tokens with offsets)."""
    paragraphs: list[tuple[int, list[Word]]] = []
    offset = 0
    for paragraph in text.split("\n"):
        words: list[Word] = []
        for match in re.finditer(r"\S+", paragraph):
            words.extend(_split_run(match.group(), offset + match.start()))
        paragraphs.append((offset, words))
        offset += len(paragraph) + 1
    return paragraphs


class LayoutTarget:
    """A region text can flow into, with per-row spans precomputed.

    Rows are resolved once at construction so re-wrapping on every keystroke
    stays cheap even on a 600 dpi page.
    """

    def __init__(self, mask: np.ndarray, padding: float = 0.0, pad_y: float | None = None):
        """`padding` insets the region; `pad_y` insets it differently down the page.

        Either may be negative, which grows the region instead -- the balloon a
        letterer wants to fill is often a few pixels wider than the one the
        artwork draws, and one word falling to a second line is the whole
        difference between natural lettering and cramped lettering.
        """
        pad_x = padding
        if pad_y is None:
            pad_y = padding

        if pad_x == pad_y:
            # The original uniform case, kept exactly as it was: a disc, not the
            # square that two separable passes would give.
            if pad_x > 0:
                r = max(1, int(round(pad_x)))
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1)
                )
                mask = cv2.erode(mask, kernel)
            elif pad_x < 0:
                r = max(1, int(round(-pad_x)))
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1)
                )
                mask = cv2.dilate(mask, kernel)
        else:
            for amount, horizontal in ((pad_x, True), (pad_y, False)):
                r = int(round(abs(amount)))
                if r < 1:
                    continue
                size = (r * 2 + 1, 1) if horizontal else (1, r * 2 + 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
                mask = cv2.erode(mask, kernel) if amount > 0 else cv2.dilate(mask, kernel)

        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            self.valid = False
            return

        self.valid = True
        self.x0, self.y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()), int(ys.max())
        self.height = y1 - self.y0 + 1
        self.width = x1 - self.x0 + 1

        crop = mask[self.y0 : y1 + 1, self.x0 : x1 + 1] > 0
        self.centre_x = float(xs.mean())
        self.centre_y = float(ys.mean())

        # Per row: the runs of set pixels, in page coordinates.
        self._runs: list[list[tuple[int, int]]] = []
        widest = None
        for row in crop:
            padded = np.concatenate(([0], row.view(np.uint8), [0]))
            edges = np.diff(padded)
            starts = np.flatnonzero(edges == 1) + self.x0
            ends = np.flatnonzero(edges == -1) + self.x0
            runs = list(zip(starts.tolist(), ends.tolist()))
            self._runs.append(runs)
            for span in runs:
                if widest is None or span[1] - span[0] > widest[1] - widest[0]:
                    widest = span
        self.widest_span = widest or (self.x0, self.x0 + self.width)

    def run_at(self, y: float, pivot_x: float) -> tuple[int, int] | None:
        """The span on row `y` containing `pivot_x`, or the nearest one."""
        row = int(round(y)) - self.y0
        if not (0 <= row < len(self._runs)):
            return None
        runs = self._runs[row]
        if not runs:
            return None

        best, best_dist = None, None
        for s, e in runs:
            if s <= pivot_x < e:
                return (s, e)
            dist = s - pivot_x if pivot_x < s else pivot_x - (e - 1)
            if best_dist is None or dist < best_dist:
                best, best_dist = (s, e), dist
        return best

    def band(self, top: float, bottom: float, pivot_x: float) -> tuple[int, int] | None:
        """The width available to a line of text occupying rows `top`..`bottom`.

        Takes the narrowest span across the band so the line clears the shape at
        every height it covers, not just at its baseline.
        """
        step = max(1, int((bottom - top) / 6))
        left, right = None, None
        y = top
        while y <= bottom:
            span = self.run_at(y, pivot_x)
            if span is None:
                return None
            left = span[0] if left is None else max(left, span[0])
            right = span[1] if right is None else min(right, span[1])
            y += step

        if left is None or right is None or right <= left:
            return None
        return (left, right)


def _advance(metrics: QFontMetricsF, text: str) -> float:
    """Width of `text`. Newlines occupy source offsets but carry no width."""
    return metrics.horizontalAdvance(text.replace("\n", ""))


def _wrap(
    paragraphs: list[tuple[int, list[Word]]],
    source: str,
    target: LayoutTarget,
    metrics: QFontMetricsF,
    line_height: float,
    first_baseline: float,
    optimistic: bool = False,
) -> list[tuple[int, int, float, float]] | None:
    """Greedily wrap, measuring available width band by band.

    Returns (start, end, left, right) per line, where start..end slices
    `source`. Candidates are measured on the real slice, so the spacing the
    author typed is what gets laid out.

    In `optimistic` mode, lines that fall outside the shape borrow its widest
    span instead of failing. That is only used while searching for the line
    count: the block starts out badly positioned because its height is not known
    yet, and failing there would reject font sizes that fit perfectly well once
    the block is re-centred.
    """
    pivot = target.centre_x
    ascent, descent = metrics.ascent(), metrics.descent()
    lines: list[tuple[int, int, float, float]] = []

    def band_for(index: int) -> tuple[int, int] | None:
        baseline = first_baseline + index * line_height
        span = target.band(baseline - ascent, baseline + descent, pivot)
        if span is None and optimistic:
            return target.widest_span
        return span

    index = 0
    for paragraph_offset, words in paragraphs:
        if not words:
            band = band_for(index)
            left, right = band if band else (pivot, pivot)
            lines.append((paragraph_offset, paragraph_offset, left, right))
            index += 1
            continue

        start = words[0].start
        end: int | None = None
        i = 0
        first = 0  # index of the first token on the line being built
        while i < len(words):
            band = band_for(index)
            if band is None:
                return None
            available = band[1] - band[0]

            if _advance(metrics, source[start : words[i].end]) <= available:
                end = words[i].end
                i += 1
                continue

            if end is None:
                if not optimistic:
                    return None  # a single word is wider than the shape here

                wide = target.widest_span
                if _advance(metrics, words[i].text) <= wide[1] - wide[0]:
                    end = words[i].end
                    i += 1
                    continue

                # Wider than the shape gets anywhere. Give it its own line and
                # let it spill; never drop the word.
                lines.append((start, words[i].end, wide[0], wide[1]))
                i += 1
                index += 1
                if i < len(words):
                    start = words[i].start
                continue

            # Kinsoku: shuffle the break earlier rather than leave a closing
            # mark stranded at the start of a line or an opening one at the end.
            brk = i
            while brk > first + 1 and words[brk].text[0] in NO_LINE_START:
                brk -= 1
            while brk > first + 1 and words[brk - 1].text[-1] in NO_LINE_END:
                brk -= 1

            lines.append((start, words[brk - 1].end, band[0], band[1]))
            start = words[brk].start
            end = None
            i = brk
            first = brk
            index += 1

        if end is not None:
            band = band_for(index)
            if band is None:
                return None
            lines.append((start, end, band[0], band[1]))
            index += 1

    return lines


def layout(
    text: str,
    target: LayoutTarget,
    font: QFont,
    line_spacing: float = 1.0,
    align: str = "center",
    strict: bool = True,
) -> list[LaidLine] | None:
    """Lay `text` out inside `target`, vertically centred.

    Line count and vertical position depend on each other -- where a line sits
    decides how wide it can be, which decides how many lines there are -- so
    this iterates to a fixed point.

    With `strict` off nothing is rejected: text too big for the shape is wrapped
    to the widest part and allowed to spill. That is the fallback the editor
    uses so oversized text stays on screen instead of silently vanishing.
    """
    if not target.valid or not text.strip():
        return []

    # A tab has no sensible width in lettering, and a character the renderer
    # skips but the measurer counts would put the caret out of step with the
    # text. A space is the same length, so every source offset still lines up.
    # Typing Tab is swallowed; this catches pasted ones.
    text = text.replace("\t", " ")

    paragraphs = tokenise(text)
    metrics = QFontMetricsF(font)
    line_height = metrics.height() * line_spacing

    # Comic lettering is all-caps with no descenders, so centring the full font
    # box leaves the text visibly high. Centre the cap-height block instead.
    cap = metrics.capHeight() or metrics.ascent()

    def baseline_for(count: int) -> float:
        visual_height = (count - 1) * line_height + cap
        return target.centre_y - visual_height / 2.0 + cap

    # Settle on a line count first, tolerating a badly placed block.
    count = 1
    seen: set[int] = set()
    for _ in range(12):
        probe = _wrap(
            paragraphs, text, target, metrics, line_height, baseline_for(count), optimistic=True
        )
        if probe is None:
            return None
        if len(probe) == count:
            break
        seen.add(count)
        count = max(1, len(probe))
        if count in seen:
            # The count oscillates when a word sits right on a wrap boundary.
            # Settle on the taller layout rather than shrinking the font.
            count = max(seen | {count})
            break

    # Then lay it out for real at that height, where the shape must be obeyed.
    # A failure here usually means the block was centred for too few lines and
    # the last one landed in the narrow bottom of the shape, so grow the count
    # and re-centre rather than shrinking the font.
    wrapped = None
    for _ in range(8):
        candidate = _wrap(
            paragraphs,
            text,
            target,
            metrics,
            line_height,
            baseline_for(count),
            optimistic=not strict,
        )
        if candidate is None:
            count += 1
            continue
        wrapped = candidate
        if len(candidate) == count:
            break
        count = len(candidate)

    if wrapped is None:
        return None

    count = len(wrapped)
    first_baseline = baseline_for(count)

    if strict:
        slack = line_height * 0.25
        if first_baseline - cap < target.y0 - slack:
            return None  # taller than the shape can hold
        if first_baseline + (count - 1) * line_height > target.y0 + target.height + slack:
            return None

    # Re-measure each line where it will actually be drawn, and reject the size
    # if one no longer clears the shape there.
    ascent, descent = metrics.ascent(), metrics.descent()
    laid: list[LaidLine] = []
    for i, (start, end, _, _) in enumerate(wrapped):
        baseline = first_baseline + i * line_height

        # A line owns the whitespace trailing it, up to where the next line
        # begins, so the caret can sit inside that gap.
        raw_end = wrapped[i + 1][0] if i + 1 < len(wrapped) else len(text)
        raw = text[start : max(start, raw_end)]
        drawn = text[start:end]

        if not drawn:
            laid.append(LaidLine("", raw, start, target.centre_x, baseline))
            continue

        span = target.band(baseline - ascent, baseline + descent, target.centre_x)
        width = _advance(metrics, drawn)

        if span is None or width > span[1] - span[0]:
            if strict:
                return None
            # Spilling out of the shape: centre the line on the widest part so
            # it stays put and stays readable rather than disappearing.
            mid = sum(target.widest_span) / 2.0
            span = (mid - width / 2.0, mid + width / 2.0)
        left, right = span

        if align == "left":
            x = left
        elif align == "right":
            x = right - width
        else:
            # Centre every line on one axis, not on the middle of whatever width
            # happens to be available at its own height. A balloon is rarely
            # symmetrical -- a tail, a lopsided outline, a dent where it meets
            # another balloon -- so centring each line in its own span makes the
            # block curve away from the side that is wider, which reads as a
            # mistake even though every line is individually centred. Letterers
            # centre a block on one axis and let the line lengths be ragged.
            #
            # The shape still has the final say: a line is nudged sideways, but
            # only as far as it must go to stay inside.
            x = target.centre_x - width / 2.0
            x = max(left, min(x, right - width))
        laid.append(LaidLine(drawn, raw, start, x, baseline))
    return laid


def fit_font_size(
    text: str,
    target: LayoutTarget,
    font: QFont,
    line_spacing: float = 1.0,
    align: str = "center",
    minimum: float = 8.0,
    maximum: float = 400.0,
    max_lines: int = 0,
) -> float:
    """Largest whole-point size at which `text` still fits `target`.

    `max_lines` caps how many lines the text may break into; 0 leaves it to the
    shape. Capping it is how you ask for width instead of height: the size that
    fits is almost always larger when the words are allowed to stack, so an
    unconstrained fit stacks them, and a wide caption that wanted one line
    across gets two down the middle instead.
    """
    probe = QFont(font)
    best = minimum

    def fits(size: float) -> bool:
        probe.setPointSizeF(size)
        lines = layout(text, target, probe, line_spacing, align)
        if not lines:
            return False
        # Blank lines are the author's own paragraph breaks, not wrapping.
        return max_lines <= 0 or sum(1 for line in lines if line.text) <= max_lines

    lo, hi = minimum, maximum
    for _ in range(20):
        mid = (lo + hi) / 2.0
        if fits(mid):
            best = mid
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.5:
            break

    # Asking for fewer lines than the shape can hold at any size leaves nothing
    # to return. Fall back to the unconstrained fit rather than to 8pt: the cap
    # is a preference, and honouring it into illegibility serves nobody.
    if max_lines > 0 and not fits(best):
        return fit_font_size(text, target, font, line_spacing, align, minimum, maximum)
    return best


# -- mapping between source offsets and rendered positions ------------------
#
# These let the editor draw a caret and selection directly onto the laid-out
# text, so editing happens on the real thing rather than on an overlay that
# wraps differently.


def line_for_index(lines: list[LaidLine], index: int) -> int:
    """Index of the line holding source offset `index`.

    Ties go to the later line, so pressing Return puts the caret at the start of
    the new line rather than leaving it trailing the old one.
    """
    chosen = 0
    for i, line in enumerate(lines):
        if line.start <= index:
            chosen = i
    return chosen


def caret_x(line: LaidLine, metrics: QFontMetricsF, index: int) -> float:
    """Where the caret sits on `line` for source offset `index`.

    Measured against `raw`, so a caret parked in trailing spaces advances by the
    width of those spaces instead of sticking to the last visible glyph.
    """
    offset = max(0, min(index - line.start, len(line.raw)))
    return line.x + _advance(metrics, line.raw[:offset])


def index_at_point(
    lines: list[LaidLine], metrics: QFontMetricsF, x: float, y: float
) -> int | None:
    """Nearest source offset to a point, for click-to-place-caret."""
    if not lines:
        return None

    i = min(range(len(lines)), key=lambda k: abs(lines[k].baseline - metrics.ascent() / 2.0 - y))
    line = lines[i]

    best, best_distance = line.start, None
    for offset in range(len(line.raw) + 1):
        distance = abs(line.x + _advance(metrics, line.raw[:offset]) - x)
        if best_distance is None or distance < best_distance:
            best, best_distance = line.start + offset, distance
    return best


# -- vertical setting (tategaki) --------------------------------------------
#
# Japanese runs top-to-bottom in columns that advance right-to-left. It is not
# horizontal text turned on its side: every glyph stays upright, each occupies a
# cell of one em however wide the character is, and a handful of marks are
# rotated or shifted to a different corner of their cell.
#
# The shape fitting reuses the existing machinery by querying a transposed copy
# of the region, where "the width available at this height" answers instead as
# "the height available in this column".

# Drawn a quarter turn round: horizontal marks and brackets whose vertical forms
# run along the column rather than across it.
VERTICAL_ROTATED = set("ー〜～（）()［］[]｛｝{}〈〉《》「」『』【】〔〕〖〗…‥ｰ―—–‐−~")

# Hang in the upper right of their cell instead of sitting on the baseline.
VERTICAL_TOP_RIGHT = set("、。，．,.")


def _fill_columns(
    cells: list[tuple[int, str]], capacities: list[int]
) -> tuple[list[list[tuple[int, str]]], bool]:
    """Pour characters down each column in turn.

    Returns the filled columns and whether anything was left over. Kinsoku is
    applied as it is horizontally: a column may not begin with closing
    punctuation or a small kana, nor end with an opening bracket.
    """
    columns: list[list[tuple[int, str]]] = []
    index = 0

    for capacity in capacities:
        if index >= len(cells):
            columns.append([])
            continue

        take = 0
        while take < capacity and index + take < len(cells):
            take += 1
            if cells[index + take - 1][1] == "\n":
                break  # a hard break ends the column

        end = index + take
        if end < len(cells) and take > 1:
            while take > 1 and cells[end][1] in NO_LINE_START:
                take -= 1
                end -= 1
            while take > 1 and cells[end - 1][1] in NO_LINE_END:
                take -= 1
                end -= 1

        columns.append(cells[index:end])
        index = end

    return columns, index < len(cells)


def layout_vertical(
    text: str,
    target: LayoutTarget,
    transposed: LayoutTarget,
    font: QFont,
    line_spacing: float = 1.0,
    strict: bool = True,
) -> list[LaidLine] | None:
    """Set `text` in vertical columns inside the region."""
    if not target.valid or not transposed.valid or not text.strip():
        return []

    metrics = QFontMetricsF(font)
    em = metrics.height()
    if em <= 0:
        return None
    column_advance = em * line_spacing
    ascent = metrics.ascent()

    cells = [(i, ch) for i, ch in enumerate(text)]

    def column_centre(index: int, count: int) -> float:
        # Column 0 is the rightmost; later columns step to the left.
        return target.centre_x + ((count - 1) / 2.0 - index) * column_advance

    def spans_for(count: int):
        spans = []
        for k in range(count):
            cx = column_centre(k, count)
            span = transposed.band(
                cx - column_advance / 2.0, cx + column_advance / 2.0, target.centre_y
            )
            if span is None:
                if strict:
                    return None
                span = transposed.widest_span
            spans.append(span)
        return spans

    count = 1
    columns: list[list[tuple[int, str]]] = []
    spans = []
    for _ in range(24):
        current = spans_for(count)
        if current is None:
            return None
        capacities = [max(0, int((s[1] - s[0]) // em)) for s in current]
        if not any(capacities):
            return None
        columns, leftover = _fill_columns(cells, capacities)
        spans = current
        if not leftover:
            break
        count += 1
    else:
        return None

    # Trim unused columns and re-centre, so the block sits in the middle of the
    # balloon rather than leaving a gap on the left.
    used = sum(1 for column in columns if column)
    if used and used != count:
        current = spans_for(used)
        if current is not None:
            capacities = [max(0, int((s[1] - s[0]) // em)) for s in current]
            trimmed, leftover = _fill_columns(cells, capacities)
            if not leftover:
                count, columns, spans = used, trimmed, current

    if strict:
        slack = column_advance * 0.25
        left = column_centre(count - 1, count) - column_advance / 2.0
        right = column_centre(0, count) + column_advance / 2.0
        if left < target.x0 - slack or right > target.x0 + target.width + slack:
            return None

    laid: list[LaidLine] = []
    for k, column in enumerate(columns):
        if not column:
            continue
        cx = column_centre(k, count)
        top, bottom = spans[k]
        # The same rule as horizontal lines, with the axes swapped: every column
        # hangs from one height rather than from the middle of whatever room it
        # personally has, so the block does not wander down the page. Columns
        # still differ in length, which is what gives a round balloon its lens
        # shape -- that comes from the wrapping, not from moving them about.
        height = len(column) * em
        start_y = target.centre_y - height / 2.0
        start_y = max(top, min(start_y, bottom - height))

        glyphs = []
        for j, (source_index, ch) in enumerate(column):
            baseline = start_y + j * em + ascent
            x = cx - _advance(metrics, ch) / 2.0
            if ch in VERTICAL_TOP_RIGHT:
                x += em * 0.32
                baseline -= em * 0.42
            glyphs.append(LaidGlyph(ch, source_index, x, baseline, ch in VERTICAL_ROTATED))

        first = column[0][0]
        following = next((later[0][0] for later in columns[k + 1 :] if later), None)
        raw_end = following if following is not None else len(text)

        laid.append(
            LaidLine(
                text="".join(ch for _, ch in column),
                raw=text[first : max(first, raw_end)],
                start=first,
                x=cx - column_advance / 2.0,
                baseline=start_y + ascent,
                glyphs=glyphs,
            )
        )
    return laid


def fit_font_size_vertical(
    text: str,
    target: LayoutTarget,
    transposed: LayoutTarget,
    font: QFont,
    line_spacing: float = 1.0,
    minimum: float = 8.0,
    maximum: float = 400.0,
    max_lines: int = 0,
) -> float:
    """Largest whole-point size at which `text` still fits vertically.

    `max_lines` caps the number of columns. Set vertically the axes swap, so
    this is the mirror of the horizontal case: capping columns asks the text to
    run down rather than across.
    """
    probe = QFont(font)
    best = minimum

    def fits(size: float) -> bool:
        probe.setPointSizeF(size)
        lines = layout_vertical(text, target, transposed, probe, line_spacing)
        if not lines:
            return False
        return max_lines <= 0 or sum(1 for line in lines if line.glyphs) <= max_lines

    lo, hi = minimum, maximum
    for _ in range(20):
        mid = (lo + hi) / 2.0
        if fits(mid):
            best = mid
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.5:
            break

    if max_lines > 0 and not fits(best):
        return fit_font_size_vertical(
            text, target, transposed, font, line_spacing, minimum, maximum
        )
    return best


def caret_geometry_vertical(
    line: LaidLine, metrics: QFontMetricsF, index: int, column_width: float
) -> tuple[float, float, float, float]:
    """Caret for a vertical column: a bar lying between two cells."""
    thickness = max(1.5, metrics.height() * 0.07)
    if not line.glyphs:
        return (line.x, line.baseline, column_width, thickness)

    ascent = metrics.ascent()
    for glyph in line.glyphs:
        if index <= glyph.start:
            return (line.x, glyph.baseline - ascent, column_width, thickness)

    last = line.glyphs[-1]
    return (line.x, last.baseline - ascent + metrics.height(), column_width, thickness)


def index_at_point_vertical(
    lines: list[LaidLine], metrics: QFontMetricsF, x: float, y: float, column_width: float
) -> int | None:
    """Nearest source offset to a point, for vertical columns."""
    if not lines:
        return None

    line = min(lines, key=lambda item: abs(item.x + column_width / 2.0 - x))
    if not line.glyphs:
        return line.start

    ascent, em = metrics.ascent(), metrics.height()
    best, best_distance = line.start, None
    for glyph in line.glyphs:
        top = glyph.baseline - ascent
        for offset, edge in ((0, top), (1, top + em)):
            distance = abs(edge - y)
            if best_distance is None or distance < best_distance:
                best, best_distance = glyph.start + offset, distance
    return best
