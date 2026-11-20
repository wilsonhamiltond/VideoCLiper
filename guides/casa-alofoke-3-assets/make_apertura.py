"""Imágenes para la apertura (0:00-0:25) de 'casa-alofoke-3-participantes'.

Uso:  python make_apertura.py
Fotos opcionales: pon una foto por participante en ./caras/ con el nombre del archivo igual al 'slug'
(ej. caras/monchy-cadena.jpg). Si no hay foto se usa una silueta.
Salida: ./out/*.png a 1920x1080.
"""
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

HERE = Path(__file__).parent
CARAS = HERE / "caras"
OUT = HERE / "out"
W, H = 1920, 1080
FONTS = Path("C:/Windows/Fonts")
GOLD, RED, WHITE, BG = (255, 196, 0), (230, 30, 60), (255, 255, 255), (10, 8, 18)

PARTICIPANTES = [
    ("monchy-cadena", "MONCHY CADENA", "El veterano"),
    ("rony-alca", "RONY ALCA", "El urbano"),
    ("german-gualey", "GERMÁN GUALEY", "De limpiabotas a viral"),
    ("manuel-connect", "MANUEL CONNECT", "El del Licey"),
    ("las-mellas", "LAS MELLAS", "Dos por una"),
    ("edward-cruz", "EDWARD CRUZ", "El cantante"),
    ("londy", "LONDY", "La modelo"),
    ("nesi-pr", "NESI PR", "La boricua"),
    ("victor-beaujour", "VÍCTOR BEAUJOUR", "Las Ocurrencias de Víctor"),
    ("alondra-carvajal", "ALONDRA CARVAJAL", "La nueva inquilina"),
    ("el-tal-bororo", "EL TAL BORORO", "¿Sorpresa?"),
]


def font(size, name="impact.ttf"):
    return ImageFont.truetype(str(FONTS / name), size)


def background(seed=0, hue=RED):
    """Fondo de estudio: degradado oscuro + focos de luz + destellos."""
    rnd = random.Random(seed)
    img = Image.new("RGB", (W, H), BG)
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(glow)
    for _ in range(5):  # focos
        x = rnd.randint(0, W)
        d.polygon([(x - 40, -10), (x + 40, -10), (x + rnd.randint(-500, 500) + 260, H), (x + rnd.randint(-500, 500) - 260, H)],
                  fill=tuple(int(c * 0.35) for c in (GOLD if rnd.random() < .5 else hue)))
    glow = glow.filter(ImageFilter.GaussianBlur(60))
    img = Image.blend(img, glow, 0.9)
    d = ImageDraw.Draw(img)
    for _ in range(140):  # destellos / bokeh
        x, y, r = rnd.randint(0, W), rnd.randint(0, H), rnd.randint(1, 4)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 230, 160))
    vign = Image.new("L", (W, H), 0)
    ImageDraw.Draw(vign).ellipse([-300, -250, W + 300, H + 250], fill=255)
    vign = vign.filter(ImageFilter.GaussianBlur(220))
    return Image.composite(img, Image.new("RGB", (W, H), (0, 0, 0)), vign)


def text_c(d, xy, txt, f, fill=WHITE, stroke=8, sfill=(0, 0, 0)):
    d.text(xy, txt, font=f, fill=fill, anchor="mm", stroke_width=stroke, stroke_fill=sfill)


def silhouette(size):
    s = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(s)
    col = (40, 34, 60, 255)
    d.ellipse([size * .30, size * .14, size * .70, size * .60], fill=col)
    d.ellipse([size * .10, size * .62, size * .90, size * 1.35], fill=col)
    text_c(d, (size / 2, size * .40), "?", font(int(size * .28)), fill=GOLD, stroke=0)
    return s


def face(slug, size):
    for ext in ("jpg", "jpeg", "png", "webp"):
        p = CARAS / f"{slug}.{ext}"
        if p.exists():
            return ImageOps.fit(Image.open(p).convert("RGBA"), (size, size), centering=(.5, .35))
    return silhouette(size)


def framed(slug, size, ring=GOLD):
    """Cara en círculo con aro."""
    out = Image.new("RGBA", (size + 24, size + 24), (0, 0, 0, 0))
    d = ImageDraw.Draw(out)
    d.ellipse([0, 0, size + 23, size + 23], fill=ring)
    d.ellipse([8, 8, size + 15, size + 15], fill=(25, 20, 40))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, size - 1, size - 1], fill=255)
    out.paste(face(slug, size), (12, 12), mask)
    return out


def title_lockup(img, cy=H // 2, scale=1.0):
    """Rótulo propio del video (no es el logo oficial): 'LA CASA 3'."""
    d = ImageDraw.Draw(img)
    text_c(d, (W / 2, cy - 150 * scale), "¿QUIÉNES SON LOS DE", font(int(70 * scale)), stroke=6)
    text_c(d, (W / 2 - 90 * scale, cy + 20 * scale), "LA CASA", font(int(210 * scale)), fill=WHITE, stroke=12)
    r = int(130 * scale)
    cx = W / 2 + 470 * scale
    d.ellipse([cx - r, cy + 20 * scale - r, cx + r, cy + 20 * scale + r], fill=RED, outline=GOLD, width=int(12 * scale))
    text_c(d, (cx, cy + 25 * scale), "3", font(int(220 * scale)), fill=GOLD, stroke=0)
    d.rectangle([W / 2 - 520 * scale, cy + 160 * scale, W / 2 + 520 * scale, cy + 172 * scale], fill=GOLD)
    text_c(d, (W / 2, cy + 230 * scale), "SU HISTORIA ANTES DEL REALITY", font(int(64 * scale)), fill=GOLD, stroke=6)


def save(img, name):
    OUT.mkdir(exist_ok=True)
    img.convert("RGB").save(OUT / name, quality=95)
    print("ok", OUT / name)


def main():
    # 1. Rótulo
    img = background(1)
    title_lockup(img)
    save(img, "01-titulo.png")

    # 2. Una tarjeta-flash por participante (cortes de ~0.4 s)
    for i, (slug, name, tag) in enumerate(PARTICIPANTES, 1):
        img = background(10 + i, hue=RED if i % 2 else (40, 90, 255))
        f = framed(slug, 620)
        img.paste(f, ((W - f.width) // 2, 110), f)
        d = ImageDraw.Draw(img)
        text_c(d, (W / 2, 840), name, font(130), stroke=10)
        text_c(d, (W / 2, 960), tag.upper(), font(60), fill=GOLD, stroke=6)
        text_c(d, (150, 90), f"#{i}", font(90), fill=GOLD, stroke=6)
        save(img, f"02-flash-{i:02d}-{slug}.png")

    # 3. Mosaico de todas las caras
    img = background(50)
    d = ImageDraw.Draw(img)
    text_c(d, (W / 2, 95), "LOS INQUILINOS DE LA CASA 3", font(96), fill=GOLD, stroke=8)
    cols, size = 6, 230
    for i, (slug, name, _) in enumerate(PARTICIPANTES):
        row, col = divmod(i, cols)
        n = cols if row == 0 else len(PARTICIPANTES) - cols
        x0 = (W - n * 300) // 2 + col * 300
        y0 = 200 + row * 420
        f = framed(slug, size, ring=GOLD if i % 2 else RED)
        img.paste(f, (x0 + (300 - f.width) // 2, y0), f)
        text_c(d, (x0 + 150, y0 + size + 70), name, font(40 if len(name) < 14 else 33), stroke=5)
    save(img, "03-mosaico-caras.png")

    # 4. Cuenta regresiva
    img = background(60)
    d = ImageDraw.Draw(img)
    text_c(d, (W / 2, 200), "FALTAN", font(130), stroke=10)
    text_c(d, (W / 2, 520), "3", font(520), fill=GOLD, stroke=18, sfill=RED)
    text_c(d, (W / 2, 820), "DÍAS", font(150), stroke=10)
    text_c(d, (W / 2, 970), "23 DE NOVIEMBRE · 9:00 PM (RD) · EN VIVO 24/7", font(58), fill=GOLD, stroke=6)
    save(img, "04-faltan-3-dias.png")

    for n in (3, 2, 1):
        img = background(70 + n)
        d = ImageDraw.Draw(img)
        d.ellipse([W / 2 - 360, H / 2 - 360, W / 2 + 360, H / 2 + 360], outline=GOLD, width=22)
        text_c(d, (W / 2, H / 2 + 10), str(n), font(560), fill=WHITE, stroke=18, sfill=RED)
        save(img, f"05-conteo-{n}.png")

    img = background(80)
    title_lockup(img, cy=430, scale=.75)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([W / 2 - 560, 760, W / 2 + 560, 960], radius=40, fill=RED, outline=GOLD, width=10)
    text_c(d, (W / 2, 830), "ESTRENO 23 DE NOVIEMBRE", font(84), stroke=0)
    text_c(d, (W / 2, 910), "9:00 PM HORA DOMINICANA", font(52), fill=GOLD, stroke=0)
    save(img, "06-estreno-23-nov.png")


if __name__ == "__main__":
    main()
