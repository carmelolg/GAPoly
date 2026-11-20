# /// script
# requires-python = ">=3.10"
# dependencies = ["Pillow>=10", "numpy>=1.24"]
# ///
"""Make the works in ``site/assets/works`` light, without changing what the site shows.

The site never uses more than it publishes: images up to ``IMAGE_WIDTH`` pixels, a thinned GIF, 80 points of the error
curve. The folders of the works are in the repository, and what is not used by the site is only weight there. This
tool keeps in them what the site can use and nothing more:

* ``history.csv``: only the rows ``build_site`` uses for the chart plus the last one (the statistics). The values are
  copied as they are, so the curve and the numbers on the site do not change by a digit.
* ``final.png``: lossless WebP (``final.webp``), only if it is smaller and decodes to exactly the same pixels.
* ``original.*``: the photo shrunk to ``IMAGE_WIDTH`` wide, JPEG quality 95. The site shows it at that size anyway.
* ``evolution.gif``: first brought to the form ``build_site`` publishes (thinned, 128 colours), then written as a lossless
  animated WebP (``evolution.webp``): every frame decodes to exactly the same pixels, with the same durations; the GIF is
  replaced only if the WebP is smaller (a GIF of few frames often is not: it stays a GIF).

It is idempotent: a second run finds nothing to do. Keep the full files outside the repository (the ``--out`` folder
of gapoly): this tool replaces the ones in ``site/assets/works``.

Usage: ``python tools/optimize_works.py [--site site] [--dry-run]``
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageSequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_site  # the same limits and the same row choice as the site build

ORIGINAL_QUALITY = 95
MIN_GAIN = 0.9  # a replacement must be smaller than this fraction of the old file
ORIGINAL_KEEP_BELOW = 1_500_000  # bytes: a photo this small and not wider than the site uses is left alone


def size_of(path: Path) -> int:
    return path.stat().st_size


def replace(path: Path, data: bytes, new_path: Path | None = None) -> None:
    """Write ``data`` atomically to ``new_path`` (default ``path``), then remove ``path`` if it is another file."""
    target = new_path or path
    # On a case-insensitive file system (macOS, Windows) original.JPG and original.jpg are the same file.
    same_file = target != path and target.exists() and path.exists() and path.samefile(target)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False, suffix=".tmp") as handle:
        handle.write(data)
    os.replace(handle.name, target)
    if same_file:
        # only the case of the name changes (never delete the file just written); two steps, since a rename between
        # names that differ by case alone can be a no-op
        middle = path.with_name(path.name + ".case")
        path.rename(middle)
        middle.rename(target)
    elif target != path:
        path.unlink()


def optimize_history(path: Path, dry_run: bool) -> int | None:
    """Keep the header, the rows the chart uses and the last row. Returns the bytes saved, or None if nothing to do."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    header, rows = lines[:1], [line for line in lines[1:] if line.strip()]
    if len(rows) <= build_site.HISTORY_POINTS:
        return None
    kept = header + [rows[i] for i in build_site.history_indices(len(rows))]
    new = "".join(kept if kept[-1].endswith("\n") else kept[:-1] + [kept[-1] + "\n"]).encode()
    saved = size_of(path) - len(new)
    if not dry_run:
        replace(path, new)
    return saved


def optimize_final(path: Path, dry_run: bool) -> int | None:
    """``final.png`` -> ``final.webp`` (lossless), if smaller and pixel for pixel the same."""
    if path.suffix.lower() != ".png":
        return None
    with Image.open(path) as opened:
        if opened.mode not in ("RGB", "L"):
            return None  # transparency or a palette: not touched
        image = opened.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, "WEBP", lossless=True, quality=100, method=6)
    data = buffer.getvalue()
    if len(data) >= size_of(path) * MIN_GAIN:
        return None
    if not np.array_equal(np.asarray(image), np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))):
        build_site.warn(f"{path}: the lossless WebP differs from the PNG, left alone")
        return None
    saved = size_of(path) - len(data)
    if not dry_run:
        replace(path, data, path.with_suffix(".webp"))
    return saved


def optimize_original(path: Path, dry_run: bool) -> int | None:
    """The photo, at most ``IMAGE_WIDTH`` wide, as a JPEG of quality 95 named ``original.jpg``."""
    with Image.open(path) as opened:
        wide = opened.width > build_site.IMAGE_WIDTH
        if not wide and size_of(path) <= ORIGINAL_KEEP_BELOW:
            return None
        icc = opened.info.get("icc_profile")
        image = ImageOps.exif_transpose(opened).convert("RGB")
    if wide:
        image = image.resize((build_site.IMAGE_WIDTH, round(image.height * build_site.IMAGE_WIDTH / image.width)),
                             Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=ORIGINAL_QUALITY, optimize=True, progressive=True, subsampling=0, icc_profile=icc)
    data = buffer.getvalue()
    if len(data) >= size_of(path) * MIN_GAIN:
        return None
    saved = size_of(path) - len(data)
    if not dry_run:
        replace(path, data, path.with_name("original.jpg"))
    return saved


def optimize_gif(path: Path, dry_run: bool) -> tuple[str, int] | None:
    """The GIF as ``build_site`` publishes it, as a lossless WebP if that is smaller, else as a lighter GIF.

    Returns what was done and the bytes saved, or None if nothing is smaller.
    """
    with tempfile.TemporaryDirectory() as folder:
        light = Path(folder) / "evolution.gif"
        build_site.web_gif(path, light)  # thinned and with fewer colours, as the site would do
        frames, durations = [], []
        with Image.open(light) as gif:
            for frame in ImageSequence.Iterator(gif):
                frames.append(frame.convert("RGB"))
                durations.append(int(frame.info.get("duration", 100)))
        webp = Path(folder) / "evolution.webp"
        build_site.save_webp_animation(frames, durations, webp)
        if size_of(webp) < size_of(path) * MIN_GAIN and same_animation(webp, frames, durations):
            saved = size_of(path) - size_of(webp)
            if not dry_run:
                replace(path, webp.read_bytes(), path.with_name("evolution.webp"))
            return "gif -> lossless webp", saved
        if size_of(light) < size_of(path) * MIN_GAIN:
            saved = size_of(path) - size_of(light)
            if not dry_run:
                replace(path, light.read_bytes())
            return "gif thinned", saved
    return None


def same_animation(webp: Path, frames: list[Image.Image], durations: list[int]) -> bool:
    """The WebP has these frames, pixel for pixel, with these durations."""
    if build_site.webp_durations(webp.read_bytes()) != durations:
        return False
    with Image.open(webp) as animation:
        decoded = [np.asarray(frame.convert("RGB")) for frame in ImageSequence.Iterator(animation)]
    return len(decoded) == len(frames) and all(
        np.array_equal(a, np.asarray(b)) for a, b in zip(decoded, frames, strict=True))


def optimize(site: Path, dry_run: bool = False) -> list[tuple[Path, str, int]]:
    """Optimise every work under ``site/assets/works``. Returns ``(file, what was done, bytes saved)``."""
    done: list[tuple[Path, str, int]] = []
    works = site / "assets" / "works"
    for folder in sorted(p for p in works.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))):
        folders = [folder, *sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith(("_", ".")))]
        for directory in folders:
            for path in sorted(p for p in directory.iterdir() if p.is_file()):
                name = path.stem.lower()
                if path.name.lower() == "history.csv":
                    action, saved = "history thinned", optimize_history(path, dry_run)
                elif name in build_site.FINAL_NAMES and path.suffix.lower() in build_site.IMAGE_SUFFIXES:
                    action, saved = "png -> lossless webp", optimize_final(path, dry_run)
                elif name in build_site.ORIGINAL_NAMES and path.suffix.lower() in build_site.IMAGE_SUFFIXES:
                    action, saved = "photo resized", optimize_original(path, dry_run)
                elif name == "evolution" and path.suffix.lower() == ".gif":
                    action, saved = optimize_gif(path, dry_run) or ("", None)
                else:
                    continue
                if saved:
                    done.append((path, action, saved))
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Make the works in site/assets/works light without changing the site.")
    parser.add_argument("--site", type=Path, default=Path("site"), help="folder with the page and the works")
    parser.add_argument("--dry-run", action="store_true", help="only report what would be saved")
    args = parser.parse_args(argv)
    if not (args.site / "assets" / "works").is_dir():
        raise SystemExit(f"{args.site}/assets/works not found: is --site the site folder?")
    done = optimize(args.site, args.dry_run)
    for path, action, saved in done:
        print(f"{saved / 1048576:8.2f} MB  {action:22s} {path.relative_to(args.site / 'assets' / 'works')}")
    total = sum(saved for _, _, saved in done)
    print(f"{'would save' if args.dry_run else 'saved'} {total / 1048576:.1f} MB in {len(done)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
