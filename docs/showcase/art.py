"""Pillow renderers for the committed cover, film loop and print edition."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

NIGHT = "#112521"
DARK = "#18352e"
INK = "#182c28"
PAPER = "#f4efe3"
WHITE = "#fffaf0"
MUTED = "#58665c"
MINT = "#a9d4be"
PINE = "#31705f"
ORANGE = "#f49266"
RUST = "#a74326"
LINE = "#c7c1b1"
FONT_DIR = Path(os.environ.get("SHOWCASE_FONT_DIR", "/usr/share/fonts/truetype/dejavu"))


@lru_cache(maxsize=None)
def font(size: int, style: str = "normal") -> ImageFont.FreeTypeFont:
    filenames = {
        "normal": "DejaVuSans", "heavy": "DejaVuSans-Bold",
        "mono": "DejaVuSansMono", "serif": "DejaVuSerif-Bold",
    }
    return ImageFont.truetype(str(FONT_DIR / f"{filenames[style]}.ttf"), size)


def caption(draw, xy, text, size, colour, style="normal", anchor="lt"):
    draw.text(xy, text, fill=colour, font=font(size, style), anchor=anchor)


def fitted(image, box, photo):
    x, y, width, height = box
    image.paste(ImageOps.fit(photo, (width, height), method=Image.Resampling.LANCZOS), (x, y))


def cover(folder, photo):
    image = Image.new("RGB", (1200, 428), NIGHT)
    d = ImageDraw.Draw(image)
    d.rounded_rectangle((16, 16, 1183, 411), radius=11, outline="#547365", width=1)
    d.line((16, 57, 1183, 57), fill="#547365")
    d.line((16, 374, 1183, 374), fill="#547365")
    d.rectangle((41, 30, 50, 39), fill=ORANGE)
    caption(d, (62, 29), "GUARDIAN OF TRUTH", 16, MINT, "mono")
    caption(d, (1156, 29), "ПОЛЕВОЙ АРХИВ / ВЫПУСК 01", 16, MINT, "mono", "rt")
    fitted(image, (535, 58, 647, 316), photo)
    d.line((534, 58, 534, 374), fill="#547365")
    caption(d, (44, 100), "Слова под", 74, WHITE, "heavy")
    caption(d, (44, 184), "сверку.", 84, MINT, "heavy")
    caption(d, (48, 306), "ОДИН ХОД · ТРИ ЭКСПОНАТА", 19, WHITE, "mono")
    caption(d, (45, 389), "ХАКАТОННЫЙ ПРОТОТИП", 14, MINT, "mono")
    caption(d, (1156, 389), "ФИГ. 01 / ВООБРАЖАЕМЫЙ КОМПАРАТОР", 14, MINT, "mono", "rt")
    image.save(folder / "archive-cover.jpg", quality=88, optimize=True, progressive=True)

    image = Image.new("RGB", (720, 608), NIGHT)
    d = ImageDraw.Draw(image)
    d.rounded_rectangle((16, 16, 704, 592), radius=12, outline="#547365")
    d.rectangle((34, 34, 45, 45), fill=ORANGE)
    caption(d, (57, 31), "GUARDIAN OF TRUTH  /  ПОЛЕВОЙ АРХИВ", 19, MINT, "mono")
    caption(d, (34, 77), "Слова под", 67, WHITE, "heavy")
    caption(d, (34, 151), "сверку.", 74, MINT, "heavy")
    fitted(image, (17, 246, 687, 314), photo)
    d.line((17, 245, 704, 245), fill="#547365")
    d.line((17, 560, 704, 560), fill="#547365")
    caption(d, (33, 572), "ФИГ. 01 / ОПТИЧЕСКИЙ КОМПАРАТОР", 17, MINT, "mono")
    image.save(folder / "archive-cover-compact.jpg", quality=88, optimize=True, progressive=True)


def tickets(folder):
    entries = (
        ("audit", "01", "Разбор хода", "КОМПАРАТОР", RUST,
         '<circle cx="253" cy="83" r="43"/><circle cx="253" cy="83" r="29"/>'
         '<path d="M191 83h124m-62-62v124m-45-22 19-19m52-52 19-19"/>',
         '<circle cx="253" cy="83" r="8"/><path d="M280 83h36" stroke-width="4"/>'),
        ("xray", "02", "Рентген", "УПАКОВКА", PINE,
         '<path d="m202 61 53-24 54 24-54 24z m0 22 53 24 54-24 m-107 22 53 24 54-24 m-54-31v19"/>',
         '<path d="m202 83 53 24 54-24" fill="none" stroke-width="4"/>'),
        ("scores", "03", "Результаты", "ПЛАСТИНА", "#805b21",
         '<path d="M202 42h105v85H202zM202 71h105m-105 28h105m-70-57v85m35-85v85"/>',
         '<path d="M205 45h29v23h-29zM240 74h29v22h-29zM275 102h29v22h-29z"/>'),
    )
    for key, number, headline, tag, ink, drawing, accent in entries:
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="332" height="164" viewBox="0 0 332 164" role="img" aria-label="{number} / {headline}">
  <rect x=".5" y=".5" width="331" height="163" rx="9" fill="{PAPER}" stroke="{LINE}"/>
  <path d="M16 126h176M16 43h139" stroke="{LINE}"/>
  <rect x="16" y="15" width="31" height="21" rx="2" fill="{ink}"/>
  <g font-family="system-ui, DejaVu Sans, sans-serif">
    <text x="31.5" y="30" text-anchor="middle" font-size="13" fill="{WHITE}">{number}</text>
    <text x="16" y="88" font-size="23" font-weight="750" fill="{INK}">{headline}</text>
    <text x="16" y="147" font-size="12" letter-spacing="1.5" fill="{ink}">{tag}</text>
    <text x="302" y="147" text-anchor="end" font-size="16" fill="{ink}">↘</text>
  </g>
  <g stroke="{ink}" stroke-width="1.8" fill="none">{drawing}</g>
  <g stroke="{ink}" fill="{ink}">{accent}</g>
</svg>
'''
        (folder / f"ticket-{key}.svg").write_text(svg, encoding="utf-8")


def replay(folder, cases):
    def tool(record):
        return record["response"].split("TOOL_CALL ", 1)[1].split(":", 1)[0]

    base = Image.new("RGB", (880, 382), NIGHT)
    d = ImageDraw.Draw(base)
    d.rounded_rectangle((12, 12, 868, 370), radius=12, outline="#577365")
    caption(d, (32, 26), "01 / ПОКАДРОВАЯ СВЕРКА", 16, MINT, "mono")
    caption(d, (848, 26), "ЗАПИСЬ / CPU", 16, MINT, "mono", "rt")
    d.line((31, 53, 848, 53), fill="#577365")
    caption(d, (32, 72), "ОДИН КАТАЛОГ. ДВА ХОДА.", 27, WHITE, "heavy")
    caption(d, (32, 117), "[AVAILABLE TOOLS]   " + tool(cases[0]), 20, MINT, "mono")
    for idx, case in enumerate(cases[:2]):
        y = 160 + idx * 78
        colour = ORANGE if case["label"] else MINT
        d.rounded_rectangle((31, y, 848, y+61), radius=6, fill=DARK, outline="#5a796c")
        d.rectangle((31, y+11, 34, y+49), fill=colour)
        caption(d, (48, y+14), "AB"[idx], 25, colour, "mono")
        caption(d, (90, y+17), tool(case), 27, WHITE, "mono")
        d.line((350, y+30, 735, y+30), fill=colour, width=2)
        d.polygon(((735, y+26), (746, y+30), (735, y+34)), fill=colour)
        d.rounded_rectangle((771, y+7, 834, y+53), radius=4, fill=colour)
        caption(d, (803, y+30), str(case["label"]), 27, NIGHT, "heavy", "mm")
    caption(d, (32, 333), f'{cases[1]["signals"][0]["severity"]} / {cases[1]["signals"][0]["kind"]}', 19, ORANGE, "mono")
    caption(d, (848, 335), "АРГУМЕНТЫ БЕЗ ИЗМЕНЕНИЙ", 14, MINT, "mono", "rt")
    frames = [base]
    for step in range(27):
        image = base.copy()
        draw = ImageDraw.Draw(image)
        x = 352 + step * 15
        draw.line((x, 160, x, 299), fill="#6c9786", width=1)
        for y, colour in ((190, MINT), (268, ORANGE)):
            draw.ellipse((x-12, y-12, x+12, y+12), outline=colour, width=1)
            draw.ellipse((x-5, y-5, x+5, y+5), fill=colour)
        frames.append(image)
    movie_path = folder / "audit-replay.gif"
    frames[0].save(movie_path, save_all=True, append_images=frames[1:],
                   loop=0, duration=[1450] + [55]*26 + [900], optimize=True)
    with Image.open(movie_path) as movie:
        movie.convert("RGB").save(folder / "audit-replay-still.png", optimize=True)


def field_sheet(folder, photo, data):
    scene = data["packing"]
    cpu = data["scores"]["regimes"][-1]
    image = Image.new("RGB", (1684, 1190), PAPER)  # A4 landscape, ~144 dpi
    d = ImageDraw.Draw(image)
    d.rectangle((37, 37, 1646, 1153), outline=LINE, width=2)
    d.line((70, 124, 1614, 124), fill=LINE, width=2)
    d.rectangle((71, 70, 92, 91), fill=RUST)
    caption(d, (110, 73), "ПОЛЕВОЙ АРХИВ / ПЕЧАТНЫЙ ЛИСТ № 01", 23, PINE, "mono")
    caption(d, (1610, 73), "AI/ML  ·  HACKATHON", 23, PINE, "mono", "rt")
    caption(d, (70, 181), "Guardian", 91, INK, "serif")
    caption(d, (71, 279), "of Truth", 91, INK, "serif")
    caption(d, (77, 415), "Детектор контекстных галлюцинаций", 30, MUTED)
    caption(d, (77, 457), "LLM-агентов. Один ход, два входа: контекст", 30, MUTED)
    caption(d, (77, 499), "и ответ. На выходе метка 0 или 1.", 30, MUTED)
    fitted(image, (840, 157, 774, 406), photo)
    caption(d, (1614, 575), "ФИГ. 01 / ВООБРАЖАЕМЫЙ КОМПАРАТОР", 19, PINE, "mono", "rt")
    cols = (
        ("01 / РАЗБОР", "0  →  1", "Вызов из каталога против", "выдуманного инструмента.", RUST),
        ("02 / РЕНТГЕН", f'{scene["raw_chars"]:,} → {scene["caps"][0]["chars"]:,}'.replace(",", " "),
         "Символы в синтетическом", "журнале до и после упаковки.", PINE),
        ("03 / РЕЗУЛЬТАТЫ", f'F1 {cpu["f1"]:.3f}',
         f'Найдено {cpu["matrix"]["tp"]} из {data["scores"]["positives"]} ошибок.',
         f'Ложных тревог: {cpu["matrix"]["fp"]}.', "#805b21"),
    )
    for index, (tag, value, desc, desc2, ink) in enumerate(cols):
        x = 72 + index * 522
        d.rectangle((x, 650, x+500, 911), fill=WHITE, outline=LINE, width=2)
        d.rectangle((x, 650, x+500, 662), fill=ink)
        caption(d, (x+22, 688), tag, 23, ink, "mono")
        caption(d, (x+22, 731), value, 53, INK, "heavy")
        caption(d, (x+22, 822), desc, 26, MUTED)
        caption(d, (x+22, 859), desc2, 26, MUTED)
    d.line((70, 984, 1614, 984), fill=LINE, width=2)
    caption(d, (74, 1011), "CPU, без LLM. 46 размеченных ходов; не независимый тест и не лидерборд.", 24, MUTED)
    caption(d, (74, 1072), "github.com/overbaffe1/Guardian-of-Truth", 26, PINE, "mono")
    image.save(folder / "field-sheet.pdf", format="PDF", resolution=144.0, quality=88,
               title="Guardian of Truth — Field Archive", subject="Hackathon project field sheet")

    share = Image.new("RGB", (1280, 640), NIGHT)
    d = ImageDraw.Draw(share)
    d.rectangle((21, 21, 1258, 619), outline="#547365", width=2)
    fitted(share, (598, 80, 660, 477), photo)
    d.line((21, 79, 1258, 79), fill="#547365", width=2)
    d.line((21, 557, 1258, 557), fill="#547365", width=2)
    caption(d, (51, 41), "ПОЛЕВОЙ АРХИВ  /  ХАКАТОН", 22, MINT, "mono")
    caption(d, (48, 173), "Guardian", 86, WHITE, "serif")
    caption(d, (48, 265), "of Truth", 86, MINT, "serif")
    caption(d, (53, 407), "Контекстные галлюцинации", 31, WHITE)
    caption(d, (53, 450), "LLM-агентов", 31, WHITE)
    caption(d, (51, 578), "01 / ХОД    02 / КОНТЕКСТ    03 / МЕТРИКИ", 23, MINT, "mono")
    share.save(folder / "share-card.jpg", quality=89, optimize=True, progressive=True)


def export_art(folder, data):
    with Image.open(folder / "comparator.jpg") as source:
        photo = source.convert("RGB")
    cover(folder, photo)
    tickets(folder)
    replay(folder, data["cases"])
    field_sheet(folder, photo, data)
