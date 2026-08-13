# EasyLetterFit

A lettering tool for manga-style comics. Click a speech balloon, type the
translation, and it wraps to the balloon's real shape.

Free software under the **GNU General Public License v3.0** — see `LICENSE`, and
`THIRD-PARTY.md` for the libraries that travel with the download.

## Installing

Take the zip from **Releases**, unpack it wherever you like, and double-click
**EasyLetterFit.exe** inside. There is nothing to install and no Python to set
up — everything it needs is in that folder. Delete the folder to uninstall.

Windows may warn about an unrecognised publisher the first time, because the
program is not code-signed. **More info → Run anyway.**

To reach it more easily, right-click `EasyLetterFit.exe` → **Send to** →
**Desktop (create shortcut)**.

## Running from source

Double-click **EasyLetterFit.cmd**. No IDE, no console window.

For a shortcut somewhere handier, right-click it → **Send to** → **Desktop
(create shortcut)**. Anything after the filename is passed through, so a
shortcut can carry a folder and open straight into one comic:

```
"EasyLetterFit.cmd" "D:\scans\chapter 3"
```

The launcher finds Python itself, preferring `pythonw` so nothing flashes up.
It skips the Microsoft Store stubs under `WindowsApps`, which are not Python —
opening one opens the Store, and the program would appear to do nothing.

It opens **where you left off** — the last folder you were in, remembered
between sessions. A comic takes many sittings, and the folder the program
happens to live in is nobody's working folder. If that folder has since moved or
gone, it falls back to the program's own without complaining.

From a terminal:

```bash
python run.py [folder-of-pages]
```

A folder given here wins over the remembered one, and then becomes the
remembered one.

Requires PySide6, Pillow, numpy and OpenCV.

### When the launcher does nothing

```bash
python run.py --self-test
```

builds the window without opening it and says which Python it used and what it
found. Startup failures are reported rather than swallowed: launched from a
shortcut there is no console to print to, so a missing dependency raises a
message box naming the interpreter and the exact `pip install` line to fix it.
Run from a terminal it prints instead — a modal box would hang anything
scripted.

## How it works

**Click** a balloon to select it — clicking directly on a letter is fine, the
seed walks out to the surrounding paper. **Double-click** to edit. Everything on
the right panel applies to the selected box.

Selecting never changes the page. The original lettering stays put until you
actually edit the text, so clicking around to look at things is safe.

Under the page list is **everything lettered on this page**, in the order it was
added. Click a row to select that box, double-click to edit it. Clicking on the
page finds the *newest* box under the cursor, so a small one made early can end
up unreachable once something larger is drawn over it — the list needs no aim at
all. Picking a row scrolls the box into view only if it is not already visible,
so choosing something on screen does not shift the page under you. The list
follows everything: boxes appearing, being renamed, deleted, undone, and
abandoned empty ones vanishing.

**Balloons drawn overlapping** are one shape as far as the paper is concerned:
each outline is drawn only outside the other, so no line divides them and a
flood fills both. They are still two balloons, and clicking one gives you that
one. Two things are looked for, because there are two ways they join:

- Overlap a little and the join leaves a **waist**. Eroding the region pinches
  it apart there before either half disappears, which is what says there were
  two bodies. This is strong enough on its own to find a balloon in a region
  that would otherwise have been turned down.
- Overlap a lot and there is no waist left — eroding just walks in towards the
  bigger one's middle. What remains is the pair of **corners** where the
  outlines cross, and the cut runs between them. Both halves have to come out
  convex, which is what stops a shout balloon being cut between two of its
  spikes. This is much weaker evidence — a chin and a fringe make two corners
  too — so it may only narrow a region already accepted as a balloon, never
  promote one that was refused.

Either way the cut lands on blank paper between the two balloons, so it never
shows in the erase. A balloon's **tail** is trimmed off the text region by the
same mechanism, which is why text no longer flows into the pointer.

**One balloon holding two blocks of lettering** — a pause, an afterthought, a
second sentence set apart — is a third case, and the shape says nothing about
it. Only the writing does. The original lettering is smeared until glyphs of the
same block run together and separate blocks do not, and clicking one gives you
that block. The smear is measured in characters rather than pixels, so it works
at any page size and in either writing direction. Like the corner cut, this may
only narrow a region already accepted, never promote a refused one — the
evidence is about what is written inside, which says nothing about whether the
thing containing it was a balloon at all.

Calibrated against 41 balloons from a real project: every one stayed a single
block from a smear of 0.5 characters upward, and a balloon that genuinely holds
two came apart anywhere below 0.8, so the setting sits at 0.6 with room either
side. Below 0.4 a single block starts falling apart into its own lines.

Editing happens on the real thing. There is no overlay — you type into the
shape-fitted text exactly as it will be exported, with the caret and selection
drawn onto the rendered glyphs. Click to place the caret, drag to select,
double-click to select a word. Arrow keys, Home/End, shift-selection, undo and
IME all behave normally, because a real (invisible) text widget still owns the
cursor.

To move text without editing it: **drag** it, or **arrow keys** to nudge (4 px,
or 20 px with Shift). **Recentre text** puts it back. A moved box can be grabbed
where the text now appears, not just over its balloon.

A selected box gets **handles**. Four square corners on the text frame scale the
lettering; four round handles on the sides stretch the area it flows into; a
grip above rotates. Corners are size, sides are room.

The sides are the answer to a balloon that is *almost* the right shape — where
one more word on the line above would read naturally and the region the artwork
gives is a few pixels too narrow for it. **Text area** in the panel does the same
by number, in pixels, and **Reset** goes back to the shape of the artwork. It
only moves where text may go: the erase still follows the balloon, so stretching
can never paint outside it.

Dragging a side holds the current font size, the same way dragging a corner
does. Left fitting, extra width only lets the font grow to match and the words
break in exactly the same places — the opposite of why anyone drags a side. Held
still, the extra room does what it looks like it should. (Typing into **Text
area** leaves fitting alone, so widening there fills the balloon with bigger
text instead.)

The frame hugs the lettering rather than the balloon, because that is what the
corners act on — a balloon's shape comes from the artwork and cannot be resized.
Dragging a corner changes the font size, and turns **Fit to balloon** off as it starts, beginning
from the size the fit had chosen rather than jumping. Hold **Shift** while
rotating to snap to 15°. Handles stay the same size on screen at any zoom, and
one drag is one undo step. Rotation is also available as a number in the
inspector; clicking and caret placement follow it either way. Rotation is useful
for matching hand-lettered sound effects that run at an angle.

Turning **Fit to balloon** off keeps the size the fit had worked out, rather
than snapping back to the stored one. Taking the fit off is how you say "nearly
right, let me adjust" — the fitted size is the place you want to start from.

**Erase L R** and **Erase T B** move each edge of the *erased* area on its own,
in pixels: positive pushes an edge outward, negative pulls it in. These are
separate from the text area controls on purpose — covering the original and
placing the translation are different jobs, and the same rectangle rarely suits
both. Original lettering does not always sit inside the balloon detection found,
and pushing one edge out beats erasing a block of it by hand.

An edge may be pushed **outside the balloon**, which is the point, and which
means it will paint over artwork if pushed too far. Only the edge asked for
moves: it is done with one-sided structuring elements rather than by clipping to
a rectangle, so a spiky shout keeps its spikes. Nudging an edge never re-runs
balloon detection, so it responds immediately rather than after a couple of
hundred milliseconds.

**Max lines** caps how many lines the text may wrap onto; "as it fits" leaves it
to the shape. This is the control for width against height. Fitting picks the
biggest font that still fits, and letting words stack almost always allows a
bigger one — so a wide caption comes out as two lines down the middle rather
than one line across it. Setting 1 asks for the single line and accepts the
smaller font that comes with it. A cap no size can honour is ignored rather than
honoured into illegibility. Your own line breaks are not wrapping and are never
counted against it. Set vertically the axes swap and the cap limits columns.

Centred text hangs from **one axis**, not from the middle of whatever width is
available at each line's own height. Balloons are rarely symmetrical — a tail, a
lopsided outline, a dent where one balloon meets another — so centring each line
in its own span makes the block curve away from whichever side is wider, which
reads as a mistake even though every line is individually centred. Letterers
centre a block on one axis and let the line lengths be ragged. The shape still
has the last word: a line that would not fit on the axis is nudged sideways, but
only as far as it must go to stay inside. Vertical setting does the same with
the axes swapped.

**H** hides the lettering and shows the page exactly as it arrived, for checking
a translation against the original without undoing the work to see it. It hides
the erases and repairs as well as the text — a white patch over the original
would defeat the point. It is a way of looking, not a property of a page, so it
stays on as you move between pages, and it has nothing to do with what gets
exported. Opening the editor brings the lettering back, since typing into text
you cannot see is typing at nothing. While you are editing, `h` is the letter h.

If text is too big to fit, it spills out of the balloon and the inspector says
so. It is never hidden — losing sight of what you typed is worse than an ugly
overflow.

**Ctrl+Z / Ctrl+Y** undo and redo, per page, up to 100 steps. Related changes
collapse into one step: a slider drag or a run of arrow-key nudges undoes in a
single press, and one editing session is one step. While you are typing, Ctrl+Z
goes to the text editor's own undo instead, so it stays character-by-character
where you expect it. History is in memory only and is not saved with the
project.

Balloons are found by flooding the blank paper around your click until it meets
the outline, then filling the letter-shaped holes. Clicking somewhere that is
not a balloon — a toned panel, open background — makes a plain rectangular box
instead, sized from the inspector.

**Escape** leaves editing; a second Escape clears the selection, for looking at
the page without a blue box on it.

## Erasing the original lettering

Inside a balloon the paper is blank, so erasing is a flat fill — white on a
normal balloon, black inside an inverted one, matching whatever the box actually
covers. A **hand-drawn box** does this too rather than reconstructing anything.
You draw one exactly where the program failed to make sense of the area on its
own, and answering that by running a second guess over the same area doubles
down on what just went wrong: a wrong rebuild pastes a patch of somebody else's
texture into the page. **Method → "Rebuild background"** is one choice away when
that is the right tool. (Clicking a spot with no balloon still starts on rebuild,
because there the program has established there is artwork rather than paper.)

Over artwork it is not. Text on a panel sits on **screentone** — not grey, but a
strictly periodic grid of dots — and a periodic pattern can be continued exactly.
"Rebuild background" measures the lattice from a clean sample nearby and tiles it
back in on the same phase, so the repair is invisible. On these pages the lattice
is 5×5 px.

Where the background is neither flat nor periodic — a face, a hand, folds of
cloth — nothing can guess it well, so the tool refuses and says so rather than
smearing. When it refuses, the artwork is left untouched: the original lettering
showing through is better than a rectangle cut out of the drawing.

Two guards decide that, and neither is about density:

- **Nothing may cross the boundary.** Text you boxed is self-contained — its
  strokes start and stop inside. A jaw, a hair edge or a sleeve carries on past
  the box, so one connected shape has real area both inside and outside. That is
  the test, because a thin jawline and a line of lettering look identical by any
  measure of how much ink is present. Screentone never trips it: its dots are
  separate specks far below the size threshold.
- **A sample must be genuinely periodic** before anything is copied from it. A
  patch of linework can match tone on density alone, and copying it drags
  somebody's sleeve into the hole.

The box is a rectangle and everything inside it gets rebuilt, so place it over
tone rather than across a figure's edge — though if you do, it will refuse rather
than damage anything.

Text is wrapped to the width actually available at each height, so it follows an
ellipse or a spiky shout balloon instead of a bounding box. "Fit to balloon"
binary-searches the largest size that still fits.

Lines are slices of what you typed, never reassembled from words, so your own
spacing survives — double spaces, deliberate gaps, hard line breaks with Return.
The caret is measured against those slices too, so it moves when you press
space, including at a line break.

Double-click selects a word, triple-click the visible line. Tab is swallowed — a
tab has no meaning in a balloon — and a pasted one is treated as a space, since
a character the renderer skips but the measurer counts would put the caret out
of step with the text.

Clicking the canvas hands keyboard focus to the view, so any click while editing
explicitly hands it back to the hidden editor. Without that, the next keystroke
goes to the page instead of the text — backspace does nothing, arrow keys nudge
the box, and Page Up scrolls the document.

Up, Down, Home, End, Page Up and Page Down move over the lines you can see. The hidden editor that
receives your keystrokes wraps text to a plain rectangle, which is not where the
lines actually break on screen, so those keys are resolved against the laid-out
lines instead — Up from the middle row reaches the top row, and Home goes to the
start of the visible line. Up and Down keep your horizontal position where the
line above or below is long enough to hold it.

The outline strokes **outward only**. A centred stroke eats into the glyph as it
widens and chokes thin lettering; this fills the union of stroke and glyph, then
the glyph on top.

## Non-Latin text

A balloon filled black and lettered in white is detected too. Flooding the blank
paper from inside one would escape immediately, since the fill itself reads as
ink, so detection tries the polarity the click landed on first and the other
after. On an inverted balloon, erasing clears back to black rather than punching
a white hole, and a new box starts with white text and a black outline.

A black balloon drawn against black artwork — brush strokes, speed lines, a
panel edge — floods as one sprawling shape with all of it, and fails on form
rather than size. When that happens the balloon is prised off what it touches by
opening the region, keeping the part under the click, and dilating back to its
true edge. The opening radius comes from how thick the region is *at the click*,
not from the bounding box of the whole flood: on a balloon fused with half a page
of artwork those are wildly different numbers, and sizing from the bounding box
opens with a radius that erases the balloon along with everything else.

When detection declines and you get a plain box instead, the status bar prints
the measurements it took, so it is possible to see which assumption was wrong.

Balloons are found by **weighted evidence**, not a chain of pass/fail tests.
That distinction matters more than any single measurement. With hard gates, each
threshold has to be loose enough to admit the most extreme legitimate balloon —
a spiky shout scores 0.61 for solidity, *worse than a screentone panel at
0.665* — and that looseness is then granted to everything else. The chain ends
up only as specific as its loosest link. Artwork rarely fails any one test
badly; it is mildly wrong on several at once, which a sum can see and a
conjunction cannot.

Five measurements are taken of the flooded region and scored out of 4.0, needing
2.3:

| measurement | what it means | balloons | not balloons |
|---|---|---|---|
| ink | how much of it is lettering | 0.09–0.39 | 0.005–0.03 |
| coarseness | median hole size, relative | 0.0019–0.0051 | 0.0003–0.0025 |
| holes | separate marks inside | 23–83 | 2–21 |
| solidity | how blob-like | 0.61–0.98 | 0.49–0.91 |
| uniformity | are the marks of a size | 0.37–0.74 | 0.10–0.73 |

Ink is scored as a **band** rather than a rising ramp: too little means empty
artwork, and too much means the region is mostly holes, which is not a balloon
interior either. Left as an ever-rising score, a shape that is 90% holes earns
full marks for it — which is exactly how a head of hair got accepted.

Holes smaller than 0.05% of the region are discarded before any of this is
measured. On a scan the fill is peppered with thousands of noise holes a few
pixels across, and they swamp the median: a real balloon reported **2161 holes
whose typical size was 0.0004% of it**, when the dozen actual letters were
hundreds of times that. Every measurement above was calibrated on clean digital
exports, where black is solid black; on scanned material they collapse without
this. Clean art has no specks, so it changes nothing there.

Four hard gates survive, for things no score should be able to argue with: a
region larger than 35% of the page cannot be a balloon, and one with fewer than
four holes is not lettering. Below four, two of the measurements stop meaning
anything — with a single hole, uniformity is 1.0 by definition and coarseness is
enormous, so a stray shape with one detail inside outscores a real balloon.

Nothing here is a size threshold in disguise. A balloon on a narrow 4-koma strip
is several times the fraction of the page that the same balloon is on a full
page.

When detection declines and you get a plain box instead, the status bar prints
the score and every measurement behind it.

Japanese and Chinese are written without spaces, so word-based wrapping gives one
enormous word that can never fit a balloon. Those scripts wrap between
characters instead, with basic kinsoku shori: a line will not begin with closing
punctuation or a small kana, nor end with an opening bracket.

Text is drawn through QPainterPath, which uses only the glyphs the chosen font
has — there is no automatic per-character fallback as there would be in a label,
so a Latin comic font asked for Japanese produces a row of identical
missing-glyph boxes. When the text needs glyphs the font lacks, a covering font
is substituted (Noto Sans JP, Yu Gothic, Meiryo… and Korean tries Hangul faces
first, since several Japanese fonts carry a partial Hangul set).

### Vertical setting (tategaki)

Tick **Vertical** on a box and the text sets in columns running top-to-bottom,
advancing right-to-left — the way a Japanese balloon is actually lettered, and
what a tall narrow balloon is shaped for.

It is not horizontal text turned on its side. Every glyph stays upright, each
takes a cell of one em however wide the character is, brackets and the
long-vowel mark are drawn a quarter turn round so they run along the column
rather than across it, and 、 and 。 hang in the upper right of their cell
instead of sitting on a baseline.

Fitting to the balloon shape reuses the horizontal machinery on a transposed
copy of the region: "the width available at this height" answers instead as "the
height available in this column". Kinsoku applies down a column as it does along
a line.

Editing follows the writing direction. Up and Down step character by character,
Left and Right move between columns — Left going forward, because columns
advance leftwards. Clicking places the caret in the right cell, and the caret
itself becomes a horizontal bar lying between two characters.

## Copying the original text

**Right-click a balloon → Copy original text** reads its original lettering onto
the clipboard, for pasting into a dictionary. The menu shows **Ctrl+Shift+C**
beside it, which does the same to whatever is selected — a shortcut nobody would
ever find on their own is worth naming somewhere it will be seen. The same menu
hangs off the rows in the page's lettering list, and right-clicking bare page
still offers the area tool and the hide toggle.

**Copy text** in the toolbar turns on a drag mode for reading any area instead —
one unknown word, or text with no balloon around it. Plain Ctrl+C still belongs
to the editor.

The balloon is read through its own shape, not its bounding box: neighbouring
balloons overlap that rectangle, and their words would come back mixed in with
the ones you asked for.

This is the only place the program reads text rather than drawing it, and it is
deliberately narrow. What it reads goes to the clipboard, never into a text box.
The translation is still yours to write.

Two engines, whichever is present, manga-ocr preferred:

| | size | on comic lettering |
| --- | --- | --- |
| **Windows OCR** | ~5 MB of bindings, plus a language pack | good on a whole balloon, weaker on stylised or mixed text |
| **manga-ocr** | ~2.5 GB, `pip install manga-ocr` | far better on vertical text and furigana — **Japanese only** |

Which language to read is **Read text in** on the right-click menu, listing
whatever this machine has OCR packs for; it defaults to one of those rather than
to any fixed language. manga-ocr is preferred only when the language is
Japanese, which is the only one it knows — handed Korean or Chinese it does not
decline, it invents Japanese.

Neither is a dependency. Without one, the dialog explains what is missing and
offers an **Install now** button that fetches the Windows bindings itself.

The Windows bindings come as **seven** packages, and they do not declare each
other — `winrt-Windows.Media.Ocr` requires only `winrt-runtime`. Install the
obvious five and everything imports, the language is found, and reading then
hangs forever, because `winrt-Windows.Foundation` carries the machinery an
`await` runs on and without it a call never completes rather than failing.

Worse, uninstalling a winrt package **leaves its folder behind**, so
`import winrt.windows.foundation` keeps succeeding against a module with nothing
in it. Every import-based test passes on that install. So availability is
settled by **actually reading a picture** — a generated one, once per session —
rather than by importing things and hoping. That check is the only one that ever
caught this.

That button exists because the manual fix is a five-package `pip` line that has
to name the right interpreter, and typing it out is both tedious and easy to get
subtly wrong — installing into a second Python the program is not running under
looks exactly like not installing anything at all. So the report always names
`sys.executable`, and the install always uses it.

**Restart after installing.** Python looked for those libraries at startup and
did not find them; a session that has had them installed underneath it is not
the same as one that started with them, and reading has been seen to hang in
that state. The dialog says so.

The picture is handed to Windows as raw pixels copied into a buffer. The obvious
route — encode a PNG, push the bytes through an `InMemoryRandomAccessStream`,
let a `BitmapDecoder` read them back — is what this did first, and it hung
forever inside `DataWriter.store_async` on a real machine while working
everywhere it was tested. Every step of that round-trip existed only to undo an
encoding the code had just performed. Copying the pixels has no asynchronous
calls in it at all, so there is nothing left to wait on, and it skips a PNG
compress and decompress as well. One `await` remains, for the recognition
itself.

Reading runs on a thread of its own and gives up after twenty seconds. The
calling thread is Qt's, inside its event loop and inside a COM apartment Qt set
up for its own purposes; blocking that on an asynchronous WinRT call is the kind
of arrangement that works everywhere until it deadlocks on somebody else's
machine. A worker gets a plain multi-threaded apartment with nothing else in it,
and whatever happens the call returns — a window that stops responding until it
is killed is a far worse answer than "that took too long".

Measured on a real page with Windows OCR: a clean balloon came back exactly
right, and one with a Latin abbreviation in it managed 83% of its characters —
enough to look a word up, not enough to trust blindly.

Reading order is rebuilt from the word positions rather than taken from the
engine. Windows OCR sometimes groups vertical Japanese *across* the columns
instead of down them, which returns the right characters in an unreadable order;
regrouping by geometry — characters into columns, columns from the right — fixes
that without disturbing text that really is horizontal.

## Colour

Every *decision* is made on a greyscale copy of the page — is this a balloon, is
this flat, is it periodic, is it safe to touch. Those are all questions about
light and dark, and grey answers them. Every *pixel that lands on the page* is
taken from the colour one.

So all three ways of covering the original work in colour:

- **Clone stamp** copies what you sampled. The repair layer holds BGR plus
  alpha; it was greyscale once, on the reasonable grounds that the pages were
  black and white line art, which held right up until sampling a blue arrow
  painted grey.
- **Erase** fills with the colour the region is actually made of, taken as the
  median over the region so the lettering being covered cannot drag it. White
  paper gives white, an inverted balloon gives black, and a caption over a
  coloured panel gives that colour instead of a hole.
- **Rebuild background** reconstructs in colour, sampling tone from the colour
  page and flat areas from the surrounding ring.

On a black-and-white page this changes nothing: every monochrome balloon
measured still erases to exactly white or exactly black. A `BoxRuntime` built
without a colour page promotes the grey one, so older code paths keep working.

## Clone stamp

For everything the rebuild refuses. **Alt+click** picks a source, then paint over
what you want covered. `[` and `]` resize the brush. Strokes go to a separate
layer, so the page image is still never touched, and each stroke is one Ctrl+Z on
the same stack as text edits.

**Snap to tone** is the part worth knowing about. Cloning onto screentone with an
arbitrary offset pastes the dots out of phase with their surroundings and leaves
a visible interference patch — the copy is clean, but the lattice beats against
itself. With snapping on, the offset is rounded to a whole number of lattice
periods when you first paint, so dots land dot-on-dot and the repair disappears.
On a test stroke over the 5x5 lattice, lattice regularity went from 0.43
unsnapped to 0.58 snapped, and the seam goes from obvious to invisible.

Undo stores only the 64px tiles a stroke touched. A whole layer is ~13 MB, far
too much to keep per stroke; a typical stroke touches four to seven tiles.

## Folders

The side panel lists subfolders as well as pages, so several comics or chapters
can sit under one directory and you can move between them. Click a folder to
enter it, `..` to go back up, or **Open folder…** for somewhere else entirely.

Both lists are ordered **numerically**, not alphabetically: runs of digits in a
name are compared as numbers, so `JGM_2` comes before `JGM_10` rather than after
it. Pages are almost always numbered, and plain text order puts page 10 between
1 and 2. Names with several numbers in them sort by each in turn, so `ch2_p10`
sits between `ch2_p2` and `ch10_p2`.

Each folder is its own project, with its own `comic_translation.json` and
`repairs/`. Switching folders saves the one you are leaving, and clears the undo
history and repair layers — both are keyed by page name, which is only unique
within a folder.

The panel rescans when you come back to the window, so a folder or image you
made in Explorer appears without restarting. **Refresh** (F5) forces it. Either
way you stay on the page you were working on.

There is nothing to set up. A folder holding pages **is** a project — open it,
click a balloon, start typing. The sidecar file writes itself when you save, so
browsing a folder you decide against leaves no trail in it.

**New project…** (Ctrl+N) makes a folder here and offers to copy pages into it.
**Add pages…** imports images into the folder you are in. Both copy rather than
move, and never overwrite: a name that collides becomes `page_1.png`, and a file
already in the folder is skipped.

Exported pages are skipped when listing pages, so an export sitting beside its
source does not come back as a page of its own. Thumbnails decode a few at a
time in the background, because a chapter folder of 600 dpi pages would
otherwise freeze the window before anything appeared.

## Files

Your page images are never modified. State lives in `comic_translation.json`
beside them, manual repairs in `repairs/`, and balloon masks are not stored — they are re-derived from the
click seed against the original pixels, which is deterministic. Export writes
`<page>_translated.png`.

**Upscale** in the toolbar sets the size, applies to both **Export page…** and
**Export all**, and is remembered between sessions. It is a standalone label
rather than a word like "at" beside a button, because that reads as a qualifier
on whichever button it happens to follow. Upscaling is not one job but two: the artwork can only be
interpolated — there is no more of it than there was — while the lettering is
outlines, so at 2x it is *drawn* at 2x and is genuinely twice as sharp rather
than twice as big. That is why this scales the painter rather than enlarging a
finished export. The artwork goes through Lanczos, which holds a line-art edge
together where bilinear turns it to porridge.

Running a small source through a neural upscaler such as waifu2x still beats
this for the *artwork*, and the two combine: upscale the source, then letter the
result. Above 80 megapixels the program asks before starting, since a 4x export
of a tall page wants the better part of a gigabyte.

## Font

**No font is required and none is assumed.** Until you pick one, lettering uses
whichever of a few widely-installed faces this machine happens to have, and the
status bar says which — a stand-in, not a recommendation. Fonts that ship inside
a drawing application are often registered privately rather than with the
operating system, so Qt cannot find them by name; loading the file directly
solves that. Choosing the lettering font:

- **Load font file…** registers a `.ttf`/`.otf`/`.ttc` for this session.
- A font file sitting **in the project folder** is registered automatically when
  that folder is opened — dropping the `.ttf` next to the pages is enough.
- **Letter in this by default** adopts whatever family is selected in the list.

That choice does two jobs. New balloons are created in it, so it no longer has
to be set per bubble; and it becomes the fallback, so a box asking for a font
this machine does not have is drawn in it instead of the stand-in. Pages
lettered before the font was available therefore fix themselves.

A box only records a family of its own once one is deliberately chosen for it,
so boxes made before you picked a font are not left asking for whatever was
installed that day.

The setting is remembered in an ini file (`QSettings`, user scope), along with
the path of the font file it came from — Qt drops application fonts when the
process ends, so remembering the family name alone would silently fall back on
the next run.

A box only records a family of its own when the font list is changed with that
box selected. Adjusting any other property leaves `font_family` alone, so
today's fallback never gets baked into the saved project.

## Building the release

```bash
python -m pip install pyinstaller
python -m PyInstaller --noconfirm --clean EasyLetterFit.spec
```

The result is `dist/EasyLetterFit/`, which is what gets zipped for Releases. A
folder rather than a single file on purpose: the Qt DLLs stay replaceable, as the
LGPL asks, and it starts without unpacking itself first. The spec collects Qt's
plugins by hand and refuses to build if the platform plugin is missing — without
it the program dies on launch with no message at all.

## Licence

GNU General Public License v3.0 or later. The full text is in `LICENSE`; the
libraries bundled with the download, and what their licences ask, are listed in
`THIRD-PARTY.md`.

This program comes with absolutely no warranty. You are welcome to redistribute
it under the conditions of the GPL.

## Not built yet

- Ruby text (furigana)
