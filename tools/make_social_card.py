#!/usr/bin/env python3
"""Render the repository's GitHub social preview card.

GitHub serves a repository's social preview image (the card shown when a link to the repository is
shared on Slack, X, Discord, LinkedIn and so on) at 1280x640. The image is uploaded through
Settings -> Social preview, which has no REST API, so this script exists to make the committed
artifact reproducible rather than an unexplained binary: run it, upload the result, commit both.

    tools/make_social_card.py                 # rewrite .github/social-preview.png
    tools/make_social_card.py --out /tmp/x.png
    tools/make_social_card.py --check         # verify the committed card exists at the right size

The palette is sampled from docs/architecture.png so the card matches the diagram the README leads
with: navy #12395C, blue #1B6CA8, light blue #8FB8D8 / #EEF5FB, amber #D9A441.

Pillow is an `eval` extra (it arrives with matplotlib), not a runtime dependency, so this script
exits with an explanatory message rather than a traceback when the extra is absent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - exercised by hand, not by the suite
    sys.exit("make_social_card.py needs Pillow: `.venv/bin/pip install 'pillow>=10'`")

W, H = 1280, 640

NAVY = (18, 57, 92)  # #12395C, the card background
PANEL = (27, 78, 120)  # #1B4E78, one step lighter, for the pipeline boxes
BLUE = (27, 108, 168)  # #1B6CA8
LIGHT_BLUE = (143, 184, 216)  # #8FB8D8, borders and secondary text
PAPER = (238, 245, 251)  # #EEF5FB, primary text
AMBER = (217, 164, 65)  # #D9A441, arrows and rules
MUTED = (201, 220, 236)  # #C9DCEC, tagline

# Avenir Next ships with macOS. The fallback list keeps the script usable elsewhere; the layout is
# measured from the loaded face, so a fallback changes the look but does not overflow the canvas.
FONT_CANDIDATES = [
    ("/System/Library/Fonts/Avenir Next.ttc", 0),  # Bold
    ("/System/Library/Fonts/Avenir Next.ttc", 5),  # Medium
    ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),
]
MEDIUM_CANDIDATES = [
    ("/System/Library/Fonts/Avenir Next.ttc", 5),  # Medium
    ("/System/Library/Fonts/Avenir Next.ttc", 0),  # Bold
    ("/System/Library/Fonts/Supplemental/Arial.ttf", 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
]

# The card's copy. Each string is a claim the README makes; keep them in step if the framing moves.
WORDMARK = "laya_rag"
TAGLINE = ["Retrieval that returns evidence,", "with calibrated probabilities —", "not prose."]
CHIPS = ["HTTP API", "MCP", "CLI", "web GUI"]
FOOTER = "github.com/thread3d/laya-rag"
PIPELINE = [
    ("interfaces", "HTTP API :8077 · MCP · CLI"),
    ("hybrid recall", "BM25 + dense, fused with RRF"),
    ("Laya re-rank + gate", "calibrated answerability"),
    ("evidence bundle", "verbatim spans, cited [n]"),
]


def _font(size: int, candidates=FONT_CANDIDATES) -> ImageFont.FreeTypeFont:
    for path, index in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size, index=index)
    return ImageFont.load_default(size)


def _text_w(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    return right - left


def render() -> Image.Image:
    img = Image.new("RGB", (W, H), NAVY)
    d = ImageDraw.Draw(img)

    # Amber rule across the top: one saturated accent, and it survives dark-mode thumbnails.
    d.rectangle([0, 0, W, 6], fill=AMBER)

    pad = 64
    title_font = _font(92)
    tag_font = _font(28, MEDIUM_CANDIDATES)
    chip_font = _font(21, MEDIUM_CANDIDATES)
    foot_font = _font(19, MEDIUM_CANDIDATES)

    # The left group (wordmark, tagline, chips) is centred on the same horizontal axis as the
    # pipeline, which spans 74..528 on the right; a top-anchored column leaves dead space below.
    d.text((pad, 118), WORDMARK, font=title_font, fill=(255, 255, 255))

    y = 244
    for line in TAGLINE:
        d.text((pad, y), line, font=tag_font, fill=MUTED)
        y += 38

    # Surface chips, laid out left to right with measured widths so the row never overflows.
    y = 418
    x = pad
    for label in CHIPS:
        tw = _text_w(d, label, chip_font)
        d.rounded_rectangle([x, y, x + tw + 38, y + 50], radius=25, outline=LIGHT_BLUE, width=2)
        d.text((x + 19, y + 12), label, font=chip_font, fill=PAPER)
        x += tw + 38 + 12

    # The pipeline, echoing docs/architecture.png: four boxes, downward arrows, no edges to trace.
    bx0, bx1 = 648, W - pad
    box_h, gap = 94, 26
    by = 74
    name_font = _font(25)
    sub_font = _font(20, MEDIUM_CANDIDATES)
    for i, (name, sub) in enumerate(PIPELINE):
        last = i == len(PIPELINE) - 1
        d.rounded_rectangle(
            [bx0, by, bx1, by + box_h],
            radius=14,
            fill=PANEL,
            outline=AMBER if last else LIGHT_BLUE,
            width=3 if last else 2,
        )
        d.text((bx0 + 26, by + 20), name, font=name_font, fill=PAPER)
        d.text((bx0 + 26, by + 54), sub, font=sub_font, fill=LIGHT_BLUE)
        if not last:
            cx = (bx0 + bx1) // 2
            d.line([cx, by + box_h + 4, cx, by + box_h + gap - 8], fill=AMBER, width=3)
            tip = by + box_h + gap
            d.polygon([(cx - 7, tip - 8), (cx + 7, tip - 8), (cx, tip)], fill=AMBER)
        by += box_h + gap

    d.text((pad, H - 44), FOOTER, font=foot_font, fill=LIGHT_BLUE)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=".github/social-preview.png", type=Path)
    ap.add_argument(
        "--check",
        action="store_true",
        help="do not render; verify the committed card exists and is 1280x640",
    )
    args = ap.parse_args()

    if args.check:
        if not args.out.exists():
            print(f"MISSING {args.out}")
            return 1
        with Image.open(args.out) as im:
            size, mode = im.size, im.mode
        ok = size == (W, H)
        print(f"{args.out}: {size[0]}x{size[1]} {mode} — {'ok' if ok else f'expected {W}x{H}'}")
        return 0 if ok else 1

    img = render()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out, format="PNG", optimize=True)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KiB, {W}x{H})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
