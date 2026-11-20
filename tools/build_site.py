# /// script
# requires-python = ">=3.10"
# dependencies = ["Pillow>=10"]
# ///
"""Build the gallery site published on GitHub Pages.

Reads ``site/`` (the page, its style and script, ``site.json``, the logo and the works) and writes a folder that is
ready to publish. The works are found by scanning ``site/assets/works/<work>/``, so adding a work is only a matter
of copying its files there: the page is regenerated from what is in the folders.

Each work folder holds what ``gapoly`` writes in its ``--out`` folder, plus the photo and an optional description::

    2026-11-20_tokyo-mosaic/
      final.png        the result (required, or a GIF to take the last frame from)
      evolution.gif    how the image evolved (optional)
      history.csv      the error per generation, gives the statistics and the chart (optional)
      original.jpg     the photo the image was made from, for the before/after slider (optional)
      meta.json        title, description, technique, shape, params... (optional)

Languages: ``site/i18n/<lang>.json`` holds the texts of the page; ``site/index.html`` is a template, written once per
language (the default language, ``it``, at the root and the others in ``<lang>/``). Titles, descriptions, tags, artists and
labels in ``site.json`` and ``meta.json`` are either a plain string (the same in every language) or ``{"it": ..., "en": ...}``.

Folders whose name starts with ``_`` or ``.`` are ignored. The images are shrunk for the web: the folders in
``site/assets/works`` keep the full-size files, the published folder only gets the light ones.

Usage, one command in every case (the header above declares the Pillow dependency, so ``uv run`` and ``pipx run``
install it by themselves; inside the project's virtual environment Pillow is already there)::

    python tools/build_site.py            # build into _site/
    python tools/build_site.py --serve    # build, then open http://127.0.0.1:8000 (Ctrl+C to stop)
    uv run tools/build_site.py --serve    # same, without an environment set up
    pipx run tools/build_site.py          # what the Pages workflow runs
"""
from __future__ import annotations

import argparse
import copy
import csv
import html
import json
import math
import os
import posixpath
import re
import shutil
import struct
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageOps, ImageSequence

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
ANIMATION_SUFFIXES = (".webp", ".gif")  # the animation of a version: a lossless animated WebP wins over a GIF
FINAL_NAMES = ("final", "result")
ORIGINAL_NAMES = ("original", "source")
LOGO_SUFFIXES = (".svg", ".png", ".webp", ".jpg")
FAVICON_SUFFIXES = (".svg", ".png", ".ico")

THUMB_WIDTH = 800  # card image
IMAGE_WIDTH = 1600  # detail image and before/after slider
GIF_WIDTH = 640  # the evolution GIF is shrunk to this width...
GIF_MAX_FRAMES = 300  # ...and thinned to this many frames
GIF_KEEP_BELOW = 3_000_000  # bytes: a GIF this small is published as it is
HISTORY_POINTS = 80  # points of the error chart kept in works.json

KNOWN_SHAPES = ("polygon", "disc", "voronoi", "mosaic", "glyph", "stroke")  # the values of gapoly's --shape
HOME_WORKS = 8  # works shown in the home page; with more, a button leads to the page with all of them
WORKS_PAGE = {"it": "opere", "en": "works"}  # folder of the page with all the works, per language
DEFAULT_LANG = "it"
LANGUAGES = {  # languages the site can be built in: locale for Intl / og:locale
    "it": {"locale": "it-IT", "og": "it_IT", "name": "Italiano"},
    "en": {"locale": "en-GB", "og": "en_GB", "name": "English"},
}
SHAPE_NAMES = {
    "polygon": {"it": "Poligoni", "en": "Polygons"}, "disc": {"it": "Dischi", "en": "Discs"},
    "voronoi": {"it": "Voronoi", "en": "Voronoi"}, "mosaic": {"it": "Mosaico", "en": "Mosaic"},
    "glyph": {"it": "Caratteri", "en": "Glyphs"}, "stroke": {"it": "Pennellate", "en": "Brush strokes"},
}
TECHNIQUE_GENETIC = {"it": "Algoritmo genetico", "en": "Genetic algorithm"}
TECHNIQUE_GENERATIVE = {"it": "Arte generativa", "en": "Generative art"}
RESULT_LABEL = {"it": "Risultato", "en": "Result"}
PLACEHOLDER = re.compile(r"\{\{\{([\w.]+)\}\}\}|\{\{([\w.]+)\}\}")  # {{{key}}} is HTML, {{key}} is escaped text

DATE_PREFIX = re.compile(r"^(\d{4}-\d{2}-\d{2})[_-](.+)$")
DATE_VALUE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

DEFAULT_SITE: dict[str, Any] = {
    "title": "GAPoly",
    "tagline": {"it": "L'arte tramite algoritmi genetici", "en": "Art through genetic algorithms"},
    "description": {
        "it": "Ho istruito un algoritmo genetico a dipingere: parte dal caos, muta e si incrocia, generazione dopo generazione, finché il quadro somiglia all'originale, che sia una foto o un'opera d'arte.",
        "en": "I taught a genetic algorithm to paint: it starts from chaos, mutates and crosses, generation after generation, until the picture resembles the original, whether a photo or a work of art.",
    },
    "author": "",
    "repo_url": "https://github.com/carmelolg/GAPoly",
    "links": [],
    "about": [],
}


def warn(message: str) -> None:
    """A warning: shown as an annotation in the GitHub Actions run, plain text elsewhere."""
    print(f"::warning::{message}" if os.environ.get("GITHUB_ACTIONS") else f"warning: {message}", file=sys.stderr)


def pick(value: Any, lang: str) -> Any:
    """``value`` in ``lang``: a ``{"it": ..., "en": ...}`` is resolved (default language, then any, as a fallback)."""
    if isinstance(value, dict):
        for key in (lang, DEFAULT_LANG):
            if key in value:
                return value[key]
        return next(iter(value.values()), "")
    return value


def text_field(value: Any, default: Any = "") -> Any:
    """A text of a meta file: a string or ``{lang: string}``; anything else is replaced by ``default``."""
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict) and value and all(isinstance(v, str) for v in value.values()):
        return value
    return default


def find_file(folder: Path, names: tuple[str, ...], suffixes: tuple[str, ...]) -> Path | None:
    """The first file in ``folder`` named ``<name><suffix>`` (any case), trying the names in order."""
    files = {p.name.lower(): p for p in sorted(folder.iterdir()) if p.is_file()}
    for name in names:
        for suffix in suffixes:
            if (name + suffix) in files:
                return files[name + suffix]
    return None


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", text.lower()).strip("-.") or "work"


def title_from(name: str) -> str:
    return re.sub(r"[-_]+", " ", name).strip().capitalize()


def split_date(folder_name: str) -> tuple[str | None, str]:
    """``2026-11-20_tokyo-mosaic`` -> (``2026-11-20``, ``tokyo-mosaic``); no date prefix -> (None, whole name)."""
    match = DATE_PREFIX.match(folder_name)
    return (match.group(1), match.group(2)) if match else (None, folder_name)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        warn(f"{path}: not valid JSON ({error}), ignored")
        return {}
    if not isinstance(data, dict):
        warn(f"{path}: expected a JSON object, ignored")
        return {}
    return data


def resized(image: Image.Image, width: int) -> Image.Image:
    """``image`` as RGB, shrunk to ``width`` if it is wider (never enlarged)."""
    image = image.convert("RGB")
    if image.width > width:
        image = image.resize((width, max(1, round(image.height * width / image.width))), Image.LANCZOS)
    return image


def save_webp(image: Image.Image, path: Path, width: int, quality: int) -> tuple[int, int]:
    out = resized(image, width)
    out.save(path, "WEBP", quality=quality, method=6)
    return out.size


def last_frame(gif_path: Path) -> Image.Image:
    with Image.open(gif_path) as gif:
        gif.seek(getattr(gif, "n_frames", 1) - 1)
        return gif.convert("RGB")


def web_gif(src: Path, dst: Path) -> None:
    """Copy a small GIF, or shrink and thin a big one (500 frames weigh tens of megabytes). Running it again on its own
    output changes nothing: a frame that already has at most 128 colours is quantised to the same colours."""
    if src.stat().st_size <= GIF_KEEP_BELOW:
        shutil.copyfile(src, dst)
        return
    with Image.open(src) as gif:
        count = getattr(gif, "n_frames", 1)
        step = max(1, math.ceil(count / GIF_MAX_FRAMES))
        frames: list[Image.Image] = []
        durations: list[int] = []
        waited = 0
        for index, frame in enumerate(ImageSequence.Iterator(gif)):
            waited += int(frame.info.get("duration", 100))
            if index % step and index != count - 1:  # keep every step-th frame and always the last
                continue
            small = resized(frame, GIF_WIDTH).quantize(colors=128, dither=Image.Dither.NONE)
            frames.append(small)
            durations.append(waited)  # a dropped frame's time goes to the one kept, so the pace is the same
            waited = 0
    frames[0].save(dst, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)


def history_indices(count: int) -> list[int]:
    """Rows of a ``count``-row history kept for the chart: evenly spaced, the first and the last included."""
    if count <= HISTORY_POINTS:
        return list(range(count))
    return sorted({round(i * (count - 1) / (HISTORY_POINTS - 1)) for i in range(HISTORY_POINTS)})


def webp_durations(data: bytes) -> list[int]:
    """Duration in milliseconds of every frame of an animated WebP (Pillow does not give it back when reading)."""
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise ValueError("not a WebP file")
    durations, position = [], 12
    while position + 8 <= len(data):
        tag, size = data[position:position + 4], struct.unpack("<I", data[position + 4:position + 8])[0]
        if tag == b"ANMF":
            durations.append(int.from_bytes(data[position + 8 + 12:position + 8 + 15], "little"))
        position += 8 + size + (size & 1)
    return durations


def save_webp_animation(frames: list[Image.Image], durations: list[int], dst: Path) -> None:
    """Lossless animated WebP: every frame decodes to exactly the pixels given."""
    frames[0].save(dst, "WEBP", save_all=True, append_images=frames[1:], duration=durations, loop=0,
                   lossless=True, quality=100, method=4)


def web_webp(src: Path, dst: Path) -> None:
    """Copy an animated WebP, or thin and shrink one that has too many frames or is too wide (still lossless)."""
    durations = webp_durations(src.read_bytes())
    with Image.open(src) as animation:
        count = getattr(animation, "n_frames", 1)
        if animation.width <= GIF_WIDTH and count <= GIF_MAX_FRAMES:
            shutil.copyfile(src, dst)
            return
        step = max(1, math.ceil(count / GIF_MAX_FRAMES))
        frames: list[Image.Image] = []
        kept: list[int] = []
        waited = 0
        for index, frame in enumerate(ImageSequence.Iterator(animation)):
            waited += durations[index] if index < len(durations) else 100
            if index % step and index != count - 1:
                continue
            frames.append(resized(frame, GIF_WIDTH))
            kept.append(waited)
            waited = 0
    save_webp_animation(frames, kept, dst)


def web_animation(src: Path, folder: Path) -> Path:
    """Put the animation of a version, in the form the site publishes, in ``folder``; returns the file written."""
    if src.suffix.lower() == ".webp":
        web_webp(src, folder / "evolution.webp")
        return folder / "evolution.webp"
    web_gif(src, folder / "evolution.gif")
    return folder / "evolution.gif"


def read_history(path: Path) -> tuple[dict[str, Any], list[float]] | None:
    """Statistics and a short error curve from the ``history.csv`` written by gapoly."""
    best: list[float] = []
    last: dict[str, str] = {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                best.append(float(row["best_mse"]))
                last = row
    except (OSError, KeyError, ValueError) as error:
        warn(f"{path}: cannot read the history ({error}), ignored")
        return None
    if not best:
        return None
    picks = history_indices(len(best))
    stats = {"generations": int(last["generation"]), "best_mse": best[-1],
             "mean_mse": float(last.get("mean_mse") or best[-1]),
             "elements": int(last.get("elements") or last["polygons"])}  # old runs called the column "polygons"
    return stats, [round(best[i], 6) for i in picks]


def label_for(name: str, shape: str | None, single: bool) -> Any:
    """Name shown on the version button: the shape's name for a folder called after it, else the folder name."""
    if single:
        return SHAPE_NAMES.get(shape or "", RESULT_LABEL)
    return SHAPE_NAMES[name.lower()] if name.lower() in SHAPE_NAMES else title_from(name)


def credits_field(value: Any) -> list[dict[str, Any]]:
    """Credit lines: ``[{"label": text or {lang: text}, "url": "https://..."}]``; only http(s) links are kept."""
    credits = []
    for item in value if isinstance(value, list) else []:
        label = text_field(item.get("label")) if isinstance(item, dict) else ""
        if not label:
            continue
        credit: dict[str, Any] = {"label": label}
        url = item.get("url")
        if isinstance(url, str) and url.startswith(("https://", "http://")):
            credit["url"] = url
        credits.append(credit)
    return credits


def tags_field(value: Any) -> Any:
    """Tags: a list of strings or ``{lang: list of strings}``."""
    if isinstance(value, dict):
        return {k: [t for t in v if isinstance(t, str)] for k, v in value.items() if isinstance(v, list)}
    return [t for t in value if isinstance(t, str)] if isinstance(value, list) else []


def build_variant(source: Path, out: Path, url: str, taken: set[str],
                  inherited: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """One way of making the photo: a folder with ``final.png`` and, if present, ``evolution.gif`` and ``history.csv``.

    ``source`` is the folder holding the files. ``inherited`` is given when the files sit directly in the work
    folder (a work with a single version): it carries the meta of the work (shape, params, label).
    """
    final = find_file(source, FINAL_NAMES, IMAGE_SUFFIXES)
    gif = find_file(source, ("evolution",), ANIMATION_SUFFIXES)
    if final is None and gif is None:
        return None
    meta = {**(inherited or {}), **load_json(source / "meta.json")}
    name = "result" if inherited is not None else source.name

    slug = base = slugify(name)
    n = 2
    while slug in taken:
        slug, n = f"{base}-{n}", n + 1
    taken.add(slug)
    target = out / slug
    target.mkdir(parents=True)

    if final is not None:
        with Image.open(final) as opened:
            picture = ImageOps.exif_transpose(opened).convert("RGB")
    else:
        picture = last_frame(gif)  # type: ignore[arg-type]  # a version with only a GIF: its last frame is the result
    width, height = save_webp(picture, target / "image.webp", IMAGE_WIDTH, 88)
    save_webp(picture, target / "thumb.webp", THUMB_WIDTH, 82)

    gif_path = None  # the key keeps its old name, but the file can be a GIF or an animated WebP
    if gif is not None:
        gif_path = f"{url}/{slug}/{web_animation(gif, target).name}"
    history = read_history(source / "history.csv") if (source / "history.csv").is_file() else None

    shape = str(meta["shape"]) if meta.get("shape") else None
    if shape is None and re.split(r"[-_ ]", name.lower())[0] in KNOWN_SHAPES:
        shape = re.split(r"[-_ ]", name.lower())[0]  # a folder called "disc" or "polygon-hq"
    return {
        "id": slug,
        "label": text_field(meta.get("label"), label_for(name, shape, inherited is not None)),
        "shape": shape,
        "image": f"{url}/{slug}/image.webp",
        "thumb": f"{url}/{slug}/thumb.webp",
        "gif": gif_path,
        "width": width,
        "height": height,
        "params": meta["params"] if isinstance(meta.get("params"), dict) else {},
        "stats": history[0] if history else None,
        "history": history[1] if history else [],
    }


def cover_score(variant: dict[str, Any]) -> float | None:
    """How good a version is: the mean of its final best and mean fitness (the mse of the best image and the average
    mse of the whole population at the last generation); lower is better. None for a version without a history."""
    stats = variant["stats"]
    return None if not stats else (stats["best_mse"] + stats["mean_mse"]) / 2


def choose_cover(variants: list[dict[str, Any]], requested: Any) -> dict[str, Any]:
    """The version shown as the work's preview: the one named by ``main`` in meta.json if it exists, else the one with
    the best fitness (see ``cover_score``), else the first."""
    if requested:
        named = next((v for v in variants if v["id"] == slugify(str(requested))), None)
        if named:
            return named
    scored = [v for v in variants if cover_score(v) is not None]
    return min(scored, key=lambda v: cover_score(v)) if scored else variants[0]  # type: ignore[arg-type,return-value]


def build_work(folder: Path, out_root: Path, taken: set[str]) -> dict[str, Any] | None:
    """A work is one photo and its versions: the files in the folder itself, and one sub-folder per further version."""
    meta = load_json(folder / "meta.json")
    date_in_name, name = split_date(folder.name)

    slug = base = slugify(folder.name)
    n = 2
    while slug in taken:
        slug, n = f"{base}-{n}", n + 1
    taken.add(slug)
    url = f"works/{slug}"
    out = out_root / slug
    out.mkdir(parents=True)

    variants: list[dict[str, Any]] = []
    used: set[str] = set()
    single = build_variant(folder, out, url, used, {k: meta[k] for k in ("shape", "params", "label") if k in meta})
    if single:
        variants.append(single)
    for sub in sorted(folder.iterdir()):
        if sub.is_dir() and not sub.name.startswith(("_", ".")):
            variant = build_variant(sub, out, url, used)
            if variant:
                variants.append(variant)
    if not variants:
        warn(f"{folder.name}: no final.png (or evolution.gif) here or in a sub-folder, skipped")
        shutil.rmtree(out)
        taken.discard(slug)
        return None

    main = choose_cover(variants, meta.get("main"))
    original = None
    original_file = find_file(folder, ORIGINAL_NAMES, IMAGE_SUFFIXES)
    if original_file is not None:
        with Image.open(original_file) as opened:
            photo = ImageOps.exif_transpose(opened).convert("RGB")
        photo = ImageOps.fit(photo, (main["width"], main["height"]), Image.LANCZOS)  # same box: the slider lines up
        save_webp(photo, out / "original.webp", IMAGE_WIDTH, 88)
        original = f"{url}/original.webp"

    date = meta.get("date", date_in_name)
    if date is not None and not (isinstance(date, str) and DATE_VALUE.match(date)):
        warn(f"{folder.name}: date {date!r} is not YYYY-MM-DD, ignored")
        date = date_in_name
    generative = any(v["history"] or v["shape"] for v in variants)
    return {
        "id": slug,
        "title": text_field(meta.get("title"), title_from(name)),
        "description": text_field(meta.get("description")),
        "technique": text_field(meta.get("technique"), TECHNIQUE_GENETIC if generative else TECHNIQUE_GENERATIVE),
        "date": date,
        "tags": tags_field(meta.get("tags")),
        "artist": text_field(meta.get("artist")),  # who made the work shown (the painting, the photo), if known
        "year": str(meta.get("year", "")).strip(),  # when it was made: "1638–1639", not the date of the evolution
        "also_known_as": tags_field(meta.get("also_known_as")),
        "location": text_field(meta.get("location")),  # where the original work is, for a photo of a monument
        "credits": credits_field(meta.get("credits")),  # lines like "Photo: Name on Unsplash", with a link
        "featured": bool(meta.get("featured", False)),
        "original": original,
        "main": main["id"],
        "shapes": sorted({v["shape"] for v in variants if v["shape"]}),
        "variants": variants,
    }


def copy_static(site: Path, out: Path) -> None:
    for name in ("404.html",):
        if (site / name).is_file():
            shutil.copyfile(site / name, out / name)
    for name in ("css", "js"):
        if (site / name).is_dir():
            shutil.copytree(site / name, out / name)
    logo_dir = site / "assets" / "logo"
    if logo_dir.is_dir():
        shutil.copytree(logo_dir, out / "assets" / "logo", ignore=shutil.ignore_patterns("README*", ".*"))


def load_strings(site: Path) -> dict[str, dict[str, str]]:
    """The dictionaries in ``site/i18n``, the default language first. Missing keys fall back to the default one."""
    found: dict[str, dict[str, str]] = {}
    for lang in LANGUAGES:
        path = site / "i18n" / f"{lang}.json"
        if path.is_file():
            found[lang] = {key: str(value) for key, value in load_json(path).items()}
    if DEFAULT_LANG not in found:
        warn(f"{site}/i18n/{DEFAULT_LANG}.json not found: the page has no texts")
    base = found.pop(DEFAULT_LANG, {})
    result = {DEFAULT_LANG: base}
    for lang, strings in found.items():
        missing = sorted(set(base) - set(strings))
        if missing:
            warn(f"i18n/{lang}.json lacks {len(missing)} key(s), the {DEFAULT_LANG} text is used: {', '.join(missing[:5])}")
        result[lang] = {**base, **strings}
    return result


def localize_site(config: dict[str, Any], lang: str) -> dict[str, Any]:
    """The site settings in one language."""
    out = copy.deepcopy(config)
    for key in ("title", "tagline", "description", "author"):
        out[key] = str(pick(config.get(key, ""), lang))
    about = pick(config.get("about", []), lang)
    out["about"] = [str(pick(paragraph, lang)) for paragraph in about] if isinstance(about, list) else []
    out["links"] = [{**link, "label": str(pick(link.get("label", ""), lang))}
                    for link in config.get("links", []) if isinstance(link, dict)]
    return out


def localize_work(work: dict[str, Any], lang: str) -> dict[str, Any]:
    out = copy.deepcopy(work)
    for key in ("title", "description", "technique", "tags", "artist", "also_known_as", "location"):
        out[key] = pick(work[key], lang)
    out["credits"] = [{**credit, "label": pick(credit["label"], lang)} for credit in work["credits"]]
    for variant in out["variants"]:
        variant["label"] = pick(variant["label"], lang)
    return out


def render_page(template: str, values: dict[str, str]) -> str:
    """Fill the placeholders of the page template: ``{{key}}`` as escaped text, ``{{{key}}}`` as HTML."""
    def fill(match: re.Match[str]) -> str:
        key = match.group(1) or match.group(2)
        if key not in values:
            warn(f"index.html uses {{{{{key}}}}} but nothing defines it")
            return ""
        return values[key] if match.group(1) else html.escape(values[key], quote=True)
    return PLACEHOLDER.sub(fill, template)


def page_dir(lang: str, page: str) -> str:
    """Folder of a page relative to the site root: '' for the home in the default language, 'en/works/' ..."""
    prefix = "" if lang == DEFAULT_LANG else f"{lang}/"
    return prefix + (f"{WORKS_PAGE.get(lang, 'works')}/" if page == "works" else "")


def link_between(source_dir: str, target_dir: str) -> str:
    """Relative link from the page in ``source_dir`` to the page in ``target_dir``."""
    relative = posixpath.relpath(target_dir.rstrip("/") or ".", start=source_dir.rstrip("/") or ".")
    return "./" if relative == "." else relative + "/"


def language_links(current: str, page: str, langs: list[str], site_url: str | None) -> tuple[str, str]:
    """The language switch for the header (it leads to the same page in the other languages), and the ``<head>`` links
    (canonical, hreflang) when the site URL is known."""
    here = page_dir(current, page)
    switch = ""
    if len(langs) > 1:
        def anchor(lang: str) -> str:
            mark = ' aria-current="true"' if lang == current else ""
            return (f'<a href="{link_between(here, page_dir(lang, page))}" hreflang="{lang}" lang="{lang}" '
                    f'title="{LANGUAGES[lang]["name"]}"{mark}>{lang.upper()}</a>')
        items = "".join(anchor(lang) for lang in langs)
        switch = f'<nav class="lang" aria-label="{{lang_aria}}">{items}</nav>'
    head = ""
    if site_url:
        root = site_url.rstrip("/") + "/"
        def url(lang: str) -> str:
            return html.escape(root + page_dir(lang, page), quote=True)
        links = [f'<link rel="canonical" href="{url(current)}">', f'<meta property="og:url" content="{url(current)}">']
        if len(langs) > 1:
            links += [f'<link rel="alternate" hreflang="{lang}" href="{url(lang)}">' for lang in langs]
            links.append(f'<link rel="alternate" hreflang="x-default" href="{url(DEFAULT_LANG)}">')
        head = "\n".join(links)
    return switch, head


def build(site: Path, out: Path) -> dict[str, Any]:
    """Write the publishable site in ``out`` (emptied first) and return the data of the default language."""
    if not (site / "index.html").is_file():
        raise SystemExit(f"{site}/index.html not found: is --site the site folder?")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    copy_static(site, out)
    (out / ".nojekyll").write_text("", encoding="utf-8")

    config = {**DEFAULT_SITE, **load_json(site / "site.json")}
    logo = find_file(site / "assets" / "logo", ("logo",), LOGO_SUFFIXES) if (site / "assets" / "logo").is_dir() else None
    favicon = (find_file(site / "assets" / "logo", ("favicon",), FAVICON_SUFFIXES)
               if (site / "assets" / "logo").is_dir() else None)
    config["logo"] = f"assets/logo/{logo.name}" if logo else None
    config["favicon"] = f"assets/logo/{favicon.name}" if favicon else config["logo"]

    works_dir = site / "assets" / "works"
    works: list[dict[str, Any]] = []
    taken: set[str] = set()
    if works_dir.is_dir():
        (out / "works").mkdir()
        for folder in sorted(works_dir.iterdir()):
            if folder.is_dir() and not folder.name.startswith(("_", ".")):
                work = build_work(folder, out / "works", taken)
                if work:
                    works.append(work)
    works.sort(key=lambda w: w["id"])
    works.sort(key=lambda w: w["date"] or "", reverse=True)  # newest first; undated last, in name order

    strings = load_strings(site)
    langs = list(strings)
    templates = {"home": (site / "index.html").read_text(encoding="utf-8")}
    if (site / "works.html").is_file():  # the page with all the works
        templates["works"] = (site / "works.html").read_text(encoding="utf-8")
    partials = {path.stem: path.read_text(encoding="utf-8") for path in sorted((site / "partials").glob("*.html"))}
    site_url = config.get("url") if isinstance(config.get("url"), str) else None
    (out / "data").mkdir()
    result: dict[str, Any] = {}
    for lang in langs:
        data = {"lang": lang, "locale": LANGUAGES[lang]["locale"], "strings": strings[lang], "home_works": HOME_WORKS,
                "site": localize_site(config, lang), "works": [localize_work(w, lang) for w in works]}
        (out / "data" / f"works.{lang}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        for page, template in templates.items():
            directory = page_dir(lang, page)
            home_url = link_between(directory, page_dir(lang, "home"))
            on_home = page == "home"
            switch, head = language_links(lang, page, langs, site_url)
            title = data["site"]["title"]
            values = {
                **strings[lang], "lang": lang, "locale": LANGUAGES[lang]["locale"], "og_locale": LANGUAGES[lang]["og"],
                "base": "../" * len(PurePosixPath(directory).parts), "works_json": f"data/works.{lang}.json", "page": page,
                "home_url": home_url, "works_url": link_between(directory, page_dir(lang, "works")),
                "brand_url": "#" if on_home else home_url, "nav_works": "#opere",
                "nav_method": "#metodo" if on_home else f"{home_url}#metodo", "nav_info": "#info" if on_home else f"{home_url}#info",
                "site_title": title, "site_tagline": data["site"]["tagline"], "site_description": data["site"]["description"],
                "page_title": f"{title} — {data['site']['tagline']}" if on_home else f"{title} — {strings[lang].get('works.page.title', '')}",
                "head_links": head,
                "lang_switch": switch.replace("{lang_aria}", html.escape(strings[lang].get("lang.aria", "Language"), quote=True)),
            }
            for name, text in partials.items():
                values[f"partial.{name}"] = render_page(text, values)
            target = out / directory
            target.mkdir(parents=True, exist_ok=True)
            (target / "index.html").write_text(render_page(template, values), encoding="utf-8")
        result[lang] = data
    return result[DEFAULT_LANG]


def make_server(directory: Path, port: int) -> ThreadingHTTPServer:
    """A local web server for the built site (``port`` 0 picks a free one). Only this machine can reach it."""
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the gallery site published on GitHub Pages.")
    parser.add_argument("--site", type=Path, default=Path("site"), help="folder with the page and the works")
    parser.add_argument("--out", type=Path, default=Path("_site"), help="folder to write (emptied first)")
    parser.add_argument("--serve", action="store_true", help="after building, show the site on http://127.0.0.1:PORT")
    parser.add_argument("--port", type=int, default=8000, help="port for --serve")
    args = parser.parse_args(argv)
    data = build(args.site, args.out)
    print(f"{len(data['works'])} work(s), languages: {', '.join(sorted(Path(p).stem.split('.')[1] for p in (args.out / 'data').glob('works.*.json')))} -> {args.out}/")
    if args.serve:
        with make_server(args.out, args.port) as server:
            print(f"http://127.0.0.1:{server.server_port}  (Ctrl+C to stop; run again to rebuild)")
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
