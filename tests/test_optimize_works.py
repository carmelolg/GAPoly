"""tools/optimize_works.py: the works get lighter and the site stays exactly the same."""
import json

import build_site
import numpy as np
import optimize_works
import pytest
from PIL import Image, ImageSequence

ROWS = 3000
HISTORY = "generation,best_mse,mean_mse,elements\n" + "".join(
    f"{g},{0.05 / (g + 1):.8f},{0.06 / (g + 1):.8f},{10 + g // 100}\n" for g in range(ROWS))


@pytest.fixture
def site(tmp_path):
    root = tmp_path / "site"
    (root / "assets" / "works").mkdir(parents=True)
    (root / "index.html").write_text("<html></html>")
    return root


def make_work(site, name="2026-11-20_w", size=(1800, 1200)):
    folder = site / "assets" / "works" / name
    version = folder / "disc"
    version.mkdir(parents=True)
    rng = np.random.default_rng(1)
    flat = (rng.integers(0, 4, (size[1] // 8, size[0] // 8, 3)) * 60).astype(np.uint8)  # blocks: compresses well
    picture = Image.fromarray(flat).resize(size, Image.NEAREST)
    picture.save(version / "final.png")
    noise = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)  # a photo is not made of flat blocks
    Image.fromarray(noise).save(folder / "original.JPG", quality=98)
    (version / "history.csv").write_text(HISTORY)
    frames = [Image.new("RGB", (64, 48), (i * 4 % 256, 60, 90)) for i in range(40)]
    frames[0].save(version / "evolution.gif", save_all=True, append_images=frames[1:], duration=50, loop=0)
    return folder


def published(site, tmp_path, label):
    data = build_site.build(site, tmp_path / label)
    return json.loads(json.dumps(data["works"]))


def without_animation(works):
    """The data of the works minus the name of the animation file (a GIF before, a WebP after)."""
    out = json.loads(json.dumps(works))
    for work in out:
        for variant in work["variants"]:
            variant.pop("gif")
    return out


def frames_and_durations(path):
    with Image.open(path) as animation:
        frames = [np.asarray(f.convert("RGB")) for f in ImageSequence.Iterator(animation)]
        if str(path).endswith(".webp"):
            return frames, build_site.webp_durations(path.read_bytes())
    with Image.open(path) as animation:
        return frames, [int(f.info["duration"]) for f in ImageSequence.Iterator(animation)]


def test_the_site_data_does_not_change_by_a_digit(site, tmp_path):
    make_work(site)
    before = published(site, tmp_path, "before")
    optimize_works.optimize(site)
    after = published(site, tmp_path, "after")
    assert without_animation(after) == without_animation(before)  # statistics, the 80 points of the curve, sizes, labels
    assert before[0]["variants"][0]["stats"]["generations"] == ROWS - 1


def test_the_history_keeps_the_rows_the_chart_uses_and_the_last_one(site):
    folder = make_work(site)
    path = folder / "disc" / "history.csv"
    optimize_works.optimize(site)
    lines = path.read_text().splitlines()
    assert lines[0] == "generation,best_mse,mean_mse,elements"
    assert len(lines) == 1 + build_site.HISTORY_POINTS
    assert lines[1].startswith("0,") and lines[-1].startswith(f"{ROWS - 1},")
    assert path.stat().st_size < 10_000


def test_final_png_becomes_a_lossless_webp_with_the_same_pixels(site):
    folder = make_work(site)
    original = np.asarray(Image.open(folder / "disc" / "final.png").convert("RGB"))
    optimize_works.optimize(site)
    assert not (folder / "disc" / "final.png").exists() and (folder / "disc" / "final.webp").is_file()
    assert np.array_equal(np.asarray(Image.open(folder / "disc" / "final.webp").convert("RGB")), original)


def test_a_png_with_transparency_is_left_alone(site):
    folder = make_work(site)
    Image.new("RGBA", (200, 200), (10, 20, 30, 128)).save(folder / "disc" / "final.png")
    optimize_works.optimize(site)
    assert (folder / "disc" / "final.png").is_file() and not (folder / "disc" / "final.webp").exists()


def test_a_big_photo_is_shrunk_to_the_width_the_site_uses(site):
    folder = make_work(site)
    before = (folder / "original.JPG").stat().st_size
    optimize_works.optimize(site)
    assert sorted(p.name for p in folder.iterdir() if p.name.lower().startswith("original")) == ["original.jpg"]
    with Image.open(folder / "original.jpg") as photo:
        assert photo.size == (build_site.IMAGE_WIDTH, 1067)
    assert (folder / "original.jpg").stat().st_size < before


def test_a_small_photo_is_not_touched(site):
    folder = make_work(site)
    Image.new("RGB", (800, 600), (5, 5, 5)).save(folder / "original.JPG")
    size = (folder / "original.JPG").stat().st_size
    optimize_works.optimize(site)
    assert (folder / "original.JPG").stat().st_size == size


def test_a_gif_becomes_a_lossless_webp_with_the_same_frames_and_durations(site, tmp_path, monkeypatch):
    monkeypatch.setattr(optimize_works, "MIN_GAIN", 10)  # the tiny test GIF is already small: take the WebP anyway
    folder = make_work(site)
    before = frames_and_durations(folder / "disc" / "evolution.gif")
    optimize_works.optimize(site)
    assert not (folder / "disc" / "evolution.gif").exists() and (folder / "disc" / "evolution.webp").is_file()
    frames, durations = frames_and_durations(folder / "disc" / "evolution.webp")
    assert durations == before[1] and len(frames) == len(before[0])
    assert all(np.array_equal(a, b) for a, b in zip(frames, before[0], strict=True))


def test_a_heavy_gif_is_thinned_as_the_site_does(site, monkeypatch):
    monkeypatch.setattr(build_site, "GIF_KEEP_BELOW", 0)
    monkeypatch.setattr(build_site, "GIF_MAX_FRAMES", 10)
    monkeypatch.setattr(optimize_works, "MIN_GAIN", 10)
    folder = make_work(site)
    optimize_works.optimize(site)
    frames, durations = frames_and_durations(folder / "disc" / "evolution.webp")
    assert len(frames) <= 11 and sum(durations) == 40 * 50  # fewer frames, the same total time


def test_a_gif_that_gains_nothing_is_left_alone(site, monkeypatch):
    monkeypatch.setattr(optimize_works, "MIN_GAIN", 0.0001)
    folder = make_work(site)
    gif = (folder / "disc" / "evolution.gif").read_bytes()
    optimize_works.optimize(site)
    assert (folder / "disc" / "evolution.gif").read_bytes() == gif and not (folder / "disc" / "evolution.webp").exists()


def test_a_webp_that_is_already_there_is_not_touched(site):
    folder = make_work(site)
    frames = [Image.new("RGB", (64, 48), (i * 4, 10, 10)) for i in range(5)]
    build_site.save_webp_animation(frames, [100] * 5, folder / "disc" / "evolution.webp")
    (folder / "disc" / "evolution.gif").unlink()
    data = (folder / "disc" / "evolution.webp").read_bytes()
    optimize_works.optimize(site)
    assert (folder / "disc" / "evolution.webp").read_bytes() == data


def test_a_second_run_finds_nothing_to_do(site):
    make_work(site)
    assert optimize_works.optimize(site)
    assert optimize_works.optimize(site) == []


def test_dry_run_reports_without_changing_anything(site):
    folder = make_work(site)
    files = {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    done = optimize_works.optimize(site, dry_run=True)
    assert done and sum(saved for _, _, saved in done) > 0
    assert files == {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()}


def test_templates_and_hidden_folders_are_left_alone(site):
    template = make_work(site, "_template")
    optimize_works.optimize(site)
    assert (template / "disc" / "final.png").is_file() and (template / "original.JPG").is_file()
