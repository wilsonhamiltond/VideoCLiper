"""Viral image maker for photo posts: turns 1 or more photos into one scroll-stopping image.

Layouts (all drawn with Pillow):
  hook     one photo, full bleed, big bold text with one highlighted word (thumbnail style)
  split    two photos stacked (side by side when square), a label on each: before / after, expectation / reality, A vs B
  collage  2-4 photos in a grid with the hook text on a band across the middle
  ranking  a numbered "top N" list: a header with the hook, then one strip per photo (up to 5)
  meme     a white caption bar on top (sentence case, like a tweet screenshot), the photo(s) below
"""
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

SIZES = {"4:5": (1080, 1350), "1:1": (1080, 1080), "9:16": (1080, 1920)}
LAYOUTS = ("hook", "split", "collage", "ranking", "meme")
YELLOW, WHITE, BLACK, RED, GREEN = (255, 214, 0), (255, 255, 255), (0, 0, 0), (235, 38, 52), (22, 163, 74)
PUNCT = ".,!?:;\"'¡¿…"

_FONT_DIRS = [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts", Path("/usr/share/fonts/truetype/msttcorefonts"),
              Path("/Library/Fonts"), Path("/System/Library/Fonts/Supplemental")]


def _font_file(*names: str) -> str | None:
    for d in _FONT_DIRS:
        for n in names:
            if (d / n).exists():
                return str(d / n)
    return None


BOLD = _font_file("impact.ttf", "Impact.ttf", "arialbd.ttf", "Arial Bold.ttf")
PLAIN = _font_file("arialbd.ttf", "Arial Bold.ttf", "segoeuib.ttf", "arial.ttf")


def _font(path: str | None, size: int):
    try:
        return ImageFont.truetype(path, size) if path else ImageFont.load_default(size)
    except OSError:
        return ImageFont.load_default(size)


def _open(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def _cover(img: Image.Image, w: int, h: int) -> Image.Image:
    return ImageOps.fit(img, (max(1, w), max(1, h)), Image.LANCZOS, centering=(0.5, 0.4))


# ---------- text ----------
def _wrap(words: list[str], font, max_w: float) -> list[list[str]]:
    lines, cur = [], []
    for w in words:
        if cur and font.getlength(" ".join(cur + [w])) > max_w:
            lines.append(cur)
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(cur)
    return lines


def _fit(text: str, path, box_w: int, box_h: int, start: int, max_lines: int = 4, min_size: int = 28):
    """The biggest font size at which `text` fits in the box: (font, lines)."""
    words = text.split()
    size = start
    while True:
        f = _font(path, size)
        lines = _wrap(words, f, box_w)
        too_wide = any(f.getlength(" ".join(l)) > box_w for l in lines)
        if (len(lines) <= max_lines and len(lines) * size * 1.12 <= box_h and not too_wide) or size <= min_size:
            return f, lines
        size = max(min_size, int(size * 0.92))


def _clean(w: str) -> str:
    return w.strip(PUNCT).casefold()


def _draw_text(d: ImageDraw.ImageDraw, text: str, box, *, accent: str = "", color=WHITE, accent_color=YELLOW,
               stroke=True, path=BOLD, start=150, max_lines=4, valign="center", upper=True):
    """Centered text in a box, auto-sized, with highlighted word(s) and a black outline."""
    text = " ".join(text.split())
    if not text:
        return
    if upper:
        text = text.upper()
    x0, y0, x1, y1 = box
    f, lines = _fit(text, path, x1 - x0, y1 - y0, start, max_lines)
    size = f.size
    lh = int(size * 1.12)
    sw = max(2, size // 14) if stroke else 0
    acc = {_clean(w) for w in accent.split()}
    total = lh * len(lines)
    y = y0 + ((y1 - y0 - total) // 2 if valign == "center" else 0 if valign == "top" else (y1 - y0 - total))
    space = f.getlength(" ")
    for line in lines:
        x = x0 + (x1 - x0 - f.getlength(" ".join(line))) / 2
        for w in line:
            d.text((x, y), w, font=f, fill=accent_color if _clean(w) in acc else color, stroke_width=sw, stroke_fill=BLACK)
            x += f.getlength(w) + space
        y += lh


def _shade(img: Image.Image, top: bool, frac: float = 0.45, strength: int = 200):
    """A dark gradient at the top or bottom so white text reads on any photo."""
    w, h = img.size
    gh = int(h * frac)
    grad = Image.linear_gradient("L").resize((w, gh))  # 0 at the top -> 255 at the bottom
    if top:
        grad = ImageOps.invert(grad)
    mask = grad.point(lambda v: v * strength // 255)
    img.paste(Image.new("RGB", (w, gh), BLACK), (0, 0 if top else h - gh), mask)


def _pill(d: ImageDraw.ImageDraw, text: str, cx: int, cy: int, size: int, fill=RED, color=WHITE, left: int | None = None):
    """A rounded label centered on (cx, cy), or starting at x=`left` when given."""
    text = text.upper().strip()
    if not text:
        return
    f = _font(BOLD, size)
    tw = f.getlength(text)
    pad = size * 0.45
    if left is not None:
        cx = left + pad + tw / 2
    d.rounded_rectangle((cx - tw / 2 - pad, cy - size * 0.75, cx + tw / 2 + pad, cy + size * 0.75),
                        radius=size * 0.35, fill=fill)
    d.text((cx, cy), text, font=f, fill=color, anchor="mm")


def _handle(d: ImageDraw.ImageDraw, handle: str, w: int, h: int):
    handle = handle.strip()
    if not handle:
        return
    if not handle.startswith("@"):
        handle = "@" + handle
    d.text((w - 28, h - 26), handle, font=_font(PLAIN, 30), fill=WHITE, anchor="rs", stroke_width=3, stroke_fill=BLACK)


# ---------- layouts ----------
def _hook(imgs, W, H, o):
    canvas = _cover(imgs[0], W, H)
    top = o["position"] != "bottom"
    if o["text"]:
        _shade(canvas, top)
    d = ImageDraw.Draw(canvas)
    zone = (60, 70, W - 60, int(H * 0.36)) if top else (60, int(H * 0.62), W - 60, H - 90)
    _draw_text(d, o["text"], zone, accent=o["accent"], valign="top" if top else "bottom")
    return canvas


def _split(imgs, W, H, o):
    a = imgs[0]
    b = imgs[1] if len(imgs) > 1 else imgs[0].transpose(Image.FLIP_LEFT_RIGHT)
    canvas = Image.new("RGB", (W, H), WHITE)
    gap = 8
    side = W >= H  # square: side by side; tall: stacked
    if side:
        cw = (W - gap) // 2
        canvas.paste(_cover(a, cw, H), (0, 0))
        canvas.paste(_cover(b, W - cw - gap, H), (cw + gap, 0))
        centers = [(cw // 2, H - 90), (cw + gap + (W - cw - gap) // 2, H - 90)]
    else:
        ch = (H - gap) // 2
        canvas.paste(_cover(a, W, ch), (0, 0))
        canvas.paste(_cover(b, W, H - ch - gap), (0, ch + gap))
        centers = [(W // 2, ch - 70), (W // 2, H - 70)]
    if o["text"]:
        _shade(canvas, True, 0.4 if side else 0.28)
    d = ImageDraw.Draw(canvas)
    labels = (o["labels"] + ["", ""])[:2]
    for (cx, cy), lab, fill in zip(centers, labels, (RED, GREEN)):
        # stacked: labels in the bottom-left corner of each half, clear of the centered VS badge
        _pill(d, lab, cx, cy, 52, fill=fill, left=None if side else 36)
    if o["text"]:
        _draw_text(d, o["text"], (50, 50, W - 50, int(H * (0.3 if side else 0.22))), accent=o["accent"], start=120,
                   max_lines=3, valign="top")
    if o["vs"]:
        cx, cy = W // 2, H // 2
        d.ellipse((cx - 70, cy - 70, cx + 70, cy + 70), fill=YELLOW, outline=BLACK, width=6)
        d.text((cx, cy), "VS", font=_font(BOLD, 72), fill=BLACK, anchor="mm")
    return canvas


def _grid_cells(n, x0, y0, W, H, gap):
    if n == 1:
        return [(x0, y0, W, H)]
    h = (H - gap) // 2
    if n == 2:
        return [(x0, y0, W, h), (x0, y0 + h + gap, W, H - h - gap)]
    w = (W - gap) // 2
    bottom = [(x0, y0 + h + gap, w, H - h - gap), (x0 + w + gap, y0 + h + gap, W - w - gap, H - h - gap)]
    if n == 3:
        return [(x0, y0, W, h)] + bottom
    return [(x0, y0, w, h), (x0 + w + gap, y0, W - w - gap, h)] + bottom


def _paste_grid(canvas, imgs, x0, y0, W, H, gap=8):
    cells = _grid_cells(min(len(imgs), 4), x0, y0, W, H, gap)
    for img, (x, y, w, h) in zip(imgs, cells):
        canvas.paste(_cover(img, w, h), (x, y))
    return cells


def _collage(imgs, W, H, o):
    canvas = Image.new("RGB", (W, H), WHITE)
    cells = _paste_grid(canvas, imgs, 0, 0, W, H)
    d = ImageDraw.Draw(canvas)
    for (x, y, w, h), lab in zip(cells, o["labels"]):
        _pill(d, lab, x + w // 2, y + h - 56, 40)
    if o["text"]:
        bh = int(H * 0.2)
        y = (H - bh) // 2
        d.rectangle((0, y, W, y + bh), fill=BLACK)
        _draw_text(d, o["text"], (50, y + 14, W - 50, y + bh - 14), accent=o["accent"], stroke=False, start=110, max_lines=2)
    return canvas


def _ranking(imgs, W, H, o):
    imgs = imgs[:5]
    canvas = Image.new("RGB", (W, H), BLACK)
    d = ImageDraw.Draw(canvas)
    head = int(H * 0.17)
    _draw_text(d, o["text"] or f"TOP {len(imgs)}", (40, 20, W - 40, head - 10), accent=o["accent"], stroke=False,
               start=110, max_lines=2)
    gap, n = 6, len(imgs)
    sh = (H - head - gap * (n - 1)) // n
    for i, img in enumerate(imgs):
        y = head + i * (sh + gap)
        canvas.paste(_cover(img, W, sh), (0, y))
        r = min(56, sh // 2 - 8)
        cx, cy = 30 + r, y + sh // 2
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=YELLOW, outline=BLACK, width=5)
        d.text((cx, cy), str(i + 1), font=_font(BOLD, int(r * 1.3)), fill=BLACK, anchor="mm")
        lab = (o["labels"][i] if i < len(o["labels"]) else "").strip().upper()
        if lab:
            f, _ = _fit(lab, BOLD, W - (cx + r + 60), sh // 2, 70, max_lines=1)
            d.text((cx + r + 24, cy), lab, font=f, fill=WHITE, anchor="lm", stroke_width=max(2, f.size // 12),
                   stroke_fill=BLACK)
    return canvas


def _meme(imgs, W, H, o):
    canvas = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(canvas)
    text = o["text"]
    bar = 0
    if text:
        f, lines = _fit(text, PLAIN, W - 100, int(H * 0.32) - 80, 72, max_lines=5, min_size=34)
        bar = min(int(H * 0.32), int(len(lines) * f.size * 1.12) + 90)
        _draw_text(d, text, (50, 0, W - 50, bar), accent=o["accent"], color=BLACK, accent_color=RED, stroke=False,
                   path=PLAIN, start=f.size, max_lines=5, upper=False)
    _paste_grid(canvas, imgs, 0, bar, W, H - bar)
    return canvas


RENDER = {"hook": _hook, "split": _split, "collage": _collage, "ranking": _ranking, "meme": _meme}


def render(paths: list[Path], out: Path, *, layout: str = "hook", aspect: str = "4:5", text: str = "", accent: str = "",
           labels: list[str] | None = None, position: str = "top", handle: str = "", vs: bool = False) -> Path:
    W, H = SIZES.get(aspect, SIZES["4:5"])
    imgs = [_open(p) for p in paths]
    if not imgs:
        raise ValueError("no photos")
    opts = {"text": str(text or "").strip()[:160], "accent": str(accent or "").strip()[:60],
            "labels": [str(x or "").strip()[:40] for x in (labels or [])][:5], "position": position, "vs": bool(vs)}
    canvas = RENDER.get(layout, _hook)(imgs, W, H, opts)
    _handle(ImageDraw.Draw(canvas), str(handle or "")[:40], W, H)
    canvas.save(out, "JPEG", quality=94, subsampling=0)
    return out
