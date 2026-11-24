"""The gallery builder: tools/build_site.py turns the folders in site/assets/works into the published site."""
import json
import threading
import urllib.request

import build_site
import pytest
from PIL import Image

HISTORY = "generation,best_mse,mean_mse,elements\n" + "".join(
    f"{g},{0.05 / (g + 1):.6f},{0.06 / (g + 1):.6f},{10 + g}\n" for g in range(301))


def picture(path, size=(120, 80), color=(200, 30, 30)):
    Image.new("RGB", size, color).save(path)


def gif(path, frames=3, size=(60, 40)):
    images = [Image.new("RGB", size, (i * 80, 40, 90)) for i in range(frames)]
    images[0].save(path, save_all=True, append_images=images[1:], duration=50, loop=0)


@pytest.fixture
def site(tmp_path):
    root = tmp_path / "site"
    (root / "css").mkdir(parents=True)
    (root / "js").mkdir()
    (root / "index.html").write_text("<html></html>")
    (root / "i18n").mkdir()
    (root / "i18n" / "it.json").write_text("{}")
    (root / "css" / "style.css").write_text("body{}")
    (root / "js" / "gallery.js").write_text("")
    (root / "assets" / "works").mkdir(parents=True)
    return root


def add_work(site, name, final=True):
    """A work folder; ``final`` puts a final.png directly in it (a work with a single version)."""
    folder = site / "assets" / "works" / name
    folder.mkdir()
    if final:
        picture(folder / "final.png")
    return folder


def add_version(work, name, **kwargs):
    folder = work / name
    folder.mkdir()
    picture(folder / "final.png", **kwargs)
    return folder


# ---- one version -------------------------------------------------------------------------------------------------

def test_a_work_with_a_single_version_is_built_with_light_images_and_its_data(site, tmp_path):
    folder = add_work(site, "2026-11-20_tokyo-mosaic")
    picture(folder / "original.jpg", size=(300, 300), color=(10, 10, 200))  # another aspect: it is cropped to match
    gif(folder / "evolution.gif")
    (folder / "history.csv").write_text(HISTORY)
    (folder / "meta.json").write_text(json.dumps(
        {"title": "Tokyo", "shape": "mosaic", "tags": ["city"], "params": {"seed": 1}}))
    out = tmp_path / "_site"

    data = build_site.build(site, out)

    (work,) = data["works"]
    assert work["id"] == "2026-11-20_tokyo-mosaic" and work["title"] == "Tokyo" and work["date"] == "2026-11-20"
    assert work["technique"] == "Genetic algorithm" and work["shapes"] == ["mosaic"] and work["tags"] == ["city"]
    (variant,) = work["variants"]
    assert work["main"] == variant["id"] and variant["shape"] == "mosaic" and variant["label"] == "Mosaic"
    assert variant["params"] == {"seed": 1}
    assert variant["stats"] == {"generations": 300, "best_mse": pytest.approx(0.05 / 301, abs=1e-6),
                                "mean_mse": pytest.approx(0.06 / 301, abs=1e-6), "elements": 310}
    assert len(variant["history"]) == build_site.HISTORY_POINTS and variant["history"][0] > variant["history"][-1]
    for path in (variant["image"], variant["thumb"], variant["gif"], work["original"]):
        assert (out / path).is_file()
    with Image.open(out / work["original"]) as original, Image.open(out / variant["image"]) as image:
        assert original.size == image.size == (variant["width"], variant["height"]) == (120, 80)
    assert json.loads((out / "data" / "works.en.json").read_text()) == data  # the default language is English
    assert (out / "index.html").is_file() and (out / "css" / "style.css").is_file() and (out / ".nojekyll").is_file()


def test_big_images_are_shrunk_for_the_web_and_small_ones_kept(site, tmp_path):
    folder = add_work(site, "big", final=False)
    picture(folder / "final.png", size=(3200, 2000))
    data = build_site.build(site, tmp_path / "out")
    (variant,) = data["works"][0]["variants"]
    assert (variant["width"], variant["height"]) == (build_site.IMAGE_WIDTH, 1000)
    with Image.open(tmp_path / "out" / variant["thumb"]) as thumb:
        assert thumb.size == (build_site.THUMB_WIDTH, 500)


def test_a_big_gif_is_shrunk_and_thinned(site, tmp_path, monkeypatch):
    monkeypatch.setattr(build_site, "GIF_KEEP_BELOW", 0)
    monkeypatch.setattr(build_site, "GIF_WIDTH", 30)
    monkeypatch.setattr(build_site, "GIF_MAX_FRAMES", 4)
    folder = add_work(site, "w")
    gif(folder / "evolution.gif", frames=12, size=(60, 40))
    data = build_site.build(site, tmp_path / "out")
    with Image.open(tmp_path / "out" / data["works"][0]["variants"][0]["gif"]) as small:
        assert small.size == (30, 20) and small.n_frames <= 5  # about GIF_MAX_FRAMES, last frame kept


def test_a_work_with_only_a_gif_takes_its_last_frame(site, tmp_path):
    folder = add_work(site, "automaton", final=False)
    gif(folder / "evolution.gif")
    (folder / "meta.json").write_text(json.dumps({"technique": "Automa cellulare"}))
    (work,) = build_site.build(site, tmp_path / "out")["works"]
    (variant,) = work["variants"]
    assert work["technique"] == "Automa cellulare" and work["original"] is None and variant["stats"] is None
    assert (tmp_path / "out" / variant["image"]).is_file()


# ---- several versions --------------------------------------------------------------------------------------------

def test_every_sub_folder_with_a_result_is_a_version_of_the_same_photo(site, tmp_path):
    folder = add_work(site, "2026-11-20_tokyo", final=False)
    picture(folder / "original.jpg", size=(300, 200))
    disc = add_version(folder, "disc")
    (disc / "history.csv").write_text(HISTORY)
    (disc / "meta.json").write_text(json.dumps({"params": {"elements": 5000}}))
    add_version(folder, "mosaic")
    add_version(folder, "polygon-hq")
    add_version(folder, "my-experiment")
    (folder / "notes").mkdir()  # no result inside: not a version
    (folder / "_old").mkdir()
    picture(folder / "_old" / "final.png")  # ignored: starts with an underscore
    out = tmp_path / "out"

    (work,) = build_site.build(site, out)["works"]

    ids = [v["id"] for v in work["variants"]]
    assert ids == ["disc", "mosaic", "my-experiment", "polygon-hq"]  # by name
    by_id = {v["id"]: v for v in work["variants"]}
    assert [by_id[i]["shape"] for i in ids] == ["disc", "mosaic", None, "polygon"]  # from the folder name when it starts with a shape
    assert [by_id[i]["label"] for i in ids] == ["Discs", "Mosaic", "My experiment", "Polygon hq"]
    assert work["shapes"] == ["disc", "mosaic", "polygon"] and by_id["disc"]["params"] == {"elements": 5000}
    assert by_id["disc"]["stats"]["generations"] == 300 and by_id["mosaic"]["stats"] is None
    assert work["original"] is not None and work["main"] == "disc"  # the first, unless meta.json says otherwise
    for variant in work["variants"]:
        assert (out / variant["image"]).is_file() and (out / variant["thumb"]).is_file()


def test_the_cover_version_is_chosen_with_main_in_meta_json(site, tmp_path):
    folder = add_work(site, "w", final=False)
    add_version(folder, "disc")
    add_version(folder, "mosaic")
    (folder / "meta.json").write_text(json.dumps({"main": "mosaic"}))
    (work,) = build_site.build(site, tmp_path / "out")["works"]
    assert work["main"] == "mosaic"


def test_shape_and_label_in_the_version_meta_win_over_the_folder_name(site, tmp_path):
    folder = add_work(site, "w", final=False)
    run = add_version(folder, "run-2")
    (run / "meta.json").write_text(json.dumps({"shape": "stroke", "label": "Seconda prova"}))
    (variant,) = build_site.build(site, tmp_path / "out")["works"][0]["variants"]
    assert variant["shape"] == "stroke" and variant["label"] == "Seconda prova"


def test_files_in_the_work_folder_and_sub_folders_are_all_versions(site, tmp_path):
    folder = add_work(site, "w")  # a final.png right here
    add_version(folder, "disc")
    (work,) = build_site.build(site, tmp_path / "out")["works"]
    assert [v["label"] for v in work["variants"]] == ["Result", "Discs"]
    assert len({v["image"] for v in work["variants"]}) == 2  # no two versions write on the same file


# ---- the whole site ----------------------------------------------------------------------------------------------

def test_folders_starting_with_underscore_or_dot_are_ignored(site, tmp_path):
    add_work(site, "_template")
    add_work(site, ".hidden")
    add_work(site, "visible")
    data = build_site.build(site, tmp_path / "out")
    assert [w["id"] for w in data["works"]] == ["visible"]


def test_a_folder_without_a_result_is_skipped_with_a_warning(site, tmp_path, capsys):
    folder = add_work(site, "empty", final=False)
    (folder / "meta.json").write_text("{}")
    data = build_site.build(site, tmp_path / "out")
    assert data["works"] == [] and "empty" in capsys.readouterr().err
    assert not (tmp_path / "out" / "works" / "empty").exists()


def test_works_are_sorted_newest_first_and_undated_last(site, tmp_path):
    for name in ("2026-01-05_a", "zzz", "2026-03-01_b", "aaa"):
        add_work(site, name)
    ids = [w["id"] for w in build_site.build(site, tmp_path / "out")["works"]]
    assert ids == ["2026-03-01_b", "2026-01-05_a", "aaa", "zzz"]


def test_title_and_date_have_sensible_defaults_and_bad_values_are_ignored(site, tmp_path, capsys):
    folder = add_work(site, "2026-02-03_tre-onde-blu")
    (folder / "meta.json").write_text(json.dumps({"date": "3 febbraio"}))
    (work,) = build_site.build(site, tmp_path / "out")["works"]
    assert work["title"] == "Tre onde blu" and work["date"] == "2026-02-03" and work["technique"] == "Generative art"
    assert "date" in capsys.readouterr().err


def test_broken_meta_json_is_ignored_with_a_warning(site, tmp_path, capsys):
    folder = add_work(site, "w")
    (folder / "meta.json").write_text("{not json")
    (work,) = build_site.build(site, tmp_path / "out")["works"]
    assert work["title"] == "W" and "not valid JSON" in capsys.readouterr().err


def test_logo_favicon_and_site_settings(site, tmp_path):
    (site / "assets" / "logo").mkdir()
    picture(site / "assets" / "logo" / "logo.png")
    (site / "assets" / "logo" / "README.md").write_text("notes")
    (site / "site.json").write_text(json.dumps({"title": "Il mio lab"}))
    out = tmp_path / "out"
    config = build_site.build(site, out)["site"]
    assert config["title"] == "Il mio lab" and config["tagline"] == build_site.DEFAULT_SITE["tagline"]["en"]
    assert config["logo"] == "assets/logo/logo.png" and config["favicon"] == config["logo"]
    assert (out / "assets" / "logo" / "logo.png").is_file() and not (out / "assets" / "logo" / "README.md").exists()


def test_without_logo_or_works_the_site_still_builds(site, tmp_path):
    data = build_site.build(site, tmp_path / "out")
    assert data["works"] == [] and data["site"]["logo"] is None


def test_the_output_folder_is_emptied_first_and_the_wrong_folder_is_refused(site, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "old.txt").write_text("stale")
    build_site.build(site, out)
    assert not (out / "old.txt").exists()
    with pytest.raises(SystemExit):
        build_site.build(tmp_path / "nowhere", out)


def test_the_built_site_can_be_served_locally(site, tmp_path):
    out = tmp_path / "out"
    build_site.build(site, out)
    with build_site.make_server(out, 0) as server:  # port 0: any free port
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # no system proxy lookup (slow on macOS)
            with opener.open(f"http://127.0.0.1:{server.server_port}/data/works.it.json", timeout=5) as reply:
                assert reply.status == 200 and "works" in json.loads(reply.read())
        finally:
            server.shutdown()
            thread.join(timeout=5)


# ---- animated WebP -------------------------------------------------------------------------------------------------

def webp(path, frames=6, size=(60, 40), duration=80):
    images = [Image.new("RGB", size, (i * 30 % 256, 40, 90)) for i in range(frames)]
    build_site.save_webp_animation(images, [duration] * frames, path)


def test_an_animated_webp_is_published_as_it_is(site, tmp_path):
    folder = add_work(site, "w")
    webp(folder / "evolution.webp")
    out = tmp_path / "out"
    (variant,) = build_site.build(site, out)["works"][0]["variants"]
    assert variant["gif"].endswith("/evolution.webp")
    assert (out / variant["gif"]).read_bytes() == (folder / "evolution.webp").read_bytes()


def test_a_webp_wins_over_a_gif_in_the_same_folder(site, tmp_path):
    folder = add_work(site, "w")
    gif(folder / "evolution.gif")
    webp(folder / "evolution.webp")
    (variant,) = build_site.build(site, tmp_path / "out")["works"][0]["variants"]
    assert variant["gif"].endswith("/evolution.webp")


def test_a_webp_with_too_many_frames_is_thinned_without_changing_the_total_time(site, tmp_path, monkeypatch):
    monkeypatch.setattr(build_site, "GIF_MAX_FRAMES", 4)
    folder = add_work(site, "w")
    webp(folder / "evolution.webp", frames=12, duration=50)
    out = tmp_path / "out"
    (variant,) = build_site.build(site, out)["works"][0]["variants"]
    published = (out / variant["gif"]).read_bytes()
    durations = build_site.webp_durations(published)
    assert len(durations) <= 5 and sum(durations) == 12 * 50


def test_a_work_with_only_a_webp_animation_takes_its_last_frame(site, tmp_path):
    folder = add_work(site, "w", final=False)
    webp(folder / "evolution.webp")
    (variant,) = build_site.build(site, tmp_path / "out")["works"][0]["variants"]
    assert (tmp_path / "out" / variant["image"]).is_file()


# ---- languages -----------------------------------------------------------------------------------------------------

TEMPLATE = (
    '<html lang="{{lang}}" data-base="{{base}}" data-works="{{works_json}}"><head><title>{{site_title}} / {{site_tagline}}</title>'
    '{{{head_links}}}</head><body>{{{lang_switch}}}<h1>{{hero.title}}</h1><div>{{{raw.html}}}</div></body></html>')
DICT_IT = {"hero.title": "Ciao", "raw.html": "<b>grassetto</b>", "lang.aria": "Lingua"}
DICT_EN = {"hero.title": "Hello", "raw.html": "<b>bold</b>", "lang.aria": "Language", "only.en": "Only English"}


@pytest.fixture
def bilingual(site):
    (site / "index.html").write_text(TEMPLATE)
    (site / "i18n" / "it.json").write_text(json.dumps(DICT_IT))
    (site / "i18n" / "en.json").write_text(json.dumps(DICT_EN))
    (site / "site.json").write_text(json.dumps({"tagline": {"it": "Immagini", "en": "Images"}, "title": "gapoly"}))
    return site


def test_a_page_and_a_data_file_are_written_for_every_language(bilingual, tmp_path):
    out = tmp_path / "out"
    build_site.build(bilingual, out)
    italian, english = (out / "it" / "index.html").read_text(), (out / "index.html").read_text()
    assert 'lang="it"' in italian and 'data-base="../"' in italian and 'data-works="data/works.it.json"' in italian
    assert 'lang="en"' in english and 'data-base=""' in english and 'data-works="data/works.en.json"' in english
    assert "<h1>Ciao</h1>" in italian and "<h1>Hello</h1>" in english
    assert "<b>grassetto</b>" in italian and "<b>bold</b>" in english  # three braces: HTML, as it is
    assert "gapoly / Immagini" in italian and "gapoly / Images" in english
    assert json.loads((out / "data" / "works.en.json").read_text())["strings"]["hero.title"] == "Hello"
    assert json.loads((out / "data" / "works.it.json").read_text())["locale"] == "it-IT"
    assert (out / "css").is_dir() and not (out / "it" / "css").exists()  # static files are shared: one copy for both pages


def test_the_language_switch_links_the_pages_to_each_other(bilingual, tmp_path):
    out = tmp_path / "out"
    build_site.build(bilingual, out)
    italian, english = (out / "it" / "index.html").read_text(), (out / "index.html").read_text()
    assert 'href="./" hreflang="it"' in italian and 'href="../" hreflang="en"' in italian
    assert 'href="it/" hreflang="it"' in english and 'href="./" hreflang="en"' in english
    assert 'aria-current="true">IT<' in italian and 'aria-current="true">EN<' in english
    assert 'aria-label="Lingua"' in italian and 'aria-label="Language"' in english


def test_canonical_and_hreflang_links_need_the_site_url(bilingual, tmp_path):
    out = tmp_path / "out"
    build_site.build(bilingual, out)
    assert "hreflang=\"x-default\"" not in (out / "index.html").read_text()  # no URL known: no absolute links
    (bilingual / "site.json").write_text(json.dumps({"url": "https://example.org/lab"}))
    build_site.build(bilingual, out)
    english = (out / "index.html").read_text()
    assert '<link rel="canonical" href="https://example.org/lab/">' in english
    assert '<link rel="alternate" hreflang="it" href="https://example.org/lab/it/">' in english
    assert '<link rel="alternate" hreflang="x-default" href="https://example.org/lab/">' in english


def test_with_one_language_there_is_no_switch_and_no_language_folder(bilingual, tmp_path):
    (bilingual / "i18n" / "it.json").unlink()
    out = tmp_path / "out"
    build_site.build(bilingual, out)
    assert not (out / "it").exists() and 'class="lang"' not in (out / "index.html").read_text()


def test_missing_translations_fall_back_to_the_default_language_with_a_warning(bilingual, tmp_path, capsys):
    out = tmp_path / "out"
    build_site.build(bilingual, out)
    assert json.loads((out / "data" / "works.it.json").read_text())["strings"]["only.en"] == "Only English"
    assert "only.en" in capsys.readouterr().err


def test_a_placeholder_nobody_defines_is_reported(bilingual, tmp_path, capsys):
    (bilingual / "index.html").write_text("<p>{{nothing.here}}</p>")
    build_site.build(bilingual, tmp_path / "out")
    assert "nothing.here" in capsys.readouterr().err


def test_texts_of_works_can_be_given_per_language(bilingual, tmp_path):
    folder = add_work(bilingual, "2026-11-20_w", final=False)
    version = add_version(folder, "disc")
    (version / "meta.json").write_text(json.dumps({"label": {"it": "Cerchi", "en": "Circles"}}))
    (folder / "meta.json").write_text(json.dumps({
        "title": {"it": "Titolo", "en": "Title"}, "description": {"it": "Descrizione", "en": "Description"},
        "technique": "Custom", "tags": {"it": ["mare"], "en": ["sea"]}}))
    out = tmp_path / "out"
    build_site.build(bilingual, out)
    work = {lang: json.loads((out / "data" / f"works.{lang}.json").read_text())["works"][0] for lang in ("it", "en")}
    assert (work["it"]["title"], work["en"]["title"]) == ("Titolo", "Title")
    assert (work["it"]["description"], work["en"]["description"]) == ("Descrizione", "Description")
    assert (work["it"]["tags"], work["en"]["tags"]) == (["mare"], ["sea"])
    assert work["it"]["technique"] == work["en"]["technique"] == "Custom"  # a plain string is the same everywhere
    assert (work["it"]["variants"][0]["label"], work["en"]["variants"][0]["label"]) == ("Cerchi", "Circles")


def test_default_names_are_translated(bilingual, tmp_path):
    folder = add_work(bilingual, "w", final=False)
    add_version(folder, "disc")
    add_version(folder, "my-run")
    (folder / "disc" / "history.csv").write_text(HISTORY)
    out = tmp_path / "out"
    build_site.build(bilingual, out)
    work = {lang: json.loads((out / "data" / f"works.{lang}.json").read_text())["works"][0] for lang in ("it", "en")}
    assert [v["label"] for v in work["it"]["variants"]] == ["Dischi", "My run"]
    assert [v["label"] for v in work["en"]["variants"]] == ["Discs", "My run"]
    assert (work["it"]["technique"], work["en"]["technique"]) == ("Algoritmo genetico", "Genetic algorithm")


def test_a_text_in_one_language_only_is_used_by_every_language(bilingual, tmp_path):
    folder = add_work(bilingual, "w")
    (folder / "meta.json").write_text(json.dumps({"title": {"it": "Solo qui"}}))
    out = tmp_path / "out"
    build_site.build(bilingual, out)
    assert json.loads((out / "data" / "works.en.json").read_text())["works"][0]["title"] == "Solo qui"


# ---- the real site: dictionaries, template and script agree ---------------------------------------------------------

REAL_SITE = __import__("pathlib").Path(__file__).resolve().parent.parent / "site"


def real_strings(lang):
    return json.loads((REAL_SITE / "i18n" / f"{lang}.json").read_text(encoding="utf-8"))


def test_every_language_has_the_same_keys_and_no_empty_text():
    italian, english = real_strings("it"), real_strings("en")
    assert set(italian) == set(english), sorted(set(italian) ^ set(english))
    assert all(text.strip() for text in [*italian.values(), *english.values()])
    for key in italian:  # the same {placeholders} in both languages
        assert sorted(__import__("re").findall(r"\{(\w+)\}", italian[key])) == \
            sorted(__import__("re").findall(r"\{(\w+)\}", english[key])), key


def test_the_template_and_the_script_only_use_keys_that_exist():
    import re
    strings = real_strings("it")
    partials = sorted(path.stem for path in (REAL_SITE / "partials").glob("*.html"))
    computed = {"lang", "locale", "og_locale", "base", "works_json", "site_title", "site_tagline", "site_description",
                "head_links", "lang_switch", "page", "page_title", "home_url", "works_url", "brand_url", "nav_works",
                "nav_method", "nav_info", *(f"partial.{name}" for name in partials)}
    files = [REAL_SITE / "index.html", REAL_SITE / "works.html", *(REAL_SITE / "partials" / f"{n}.html" for n in partials)]
    used = set()
    for file in files:
        template = re.sub(r"<!--.*?-->", "", file.read_text(encoding="utf-8"), flags=re.DOTALL)
        used |= {a or b for a, b in build_site.PLACEHOLDER.findall(template)}
    assert used - computed - set(strings) == set()
    assert {"partial.head", "partial.header", "partial.footer", "partial.viewer"} <= used
    script = (REAL_SITE / "js" / "gallery.js").read_text(encoding="utf-8")
    keys = set(re.findall(r'\bt\("([\w.]+)"', script))
    for base in re.findall(r'\bplural\("([\w.]+)"', script):
        keys |= {f"{base}_one", f"{base}_other"}
    for shape in build_site.KNOWN_SHAPES:  # built from the shape name at run time
        keys |= {f"shape.{shape}", f"shape.{shape}.text"}
    for view in ("compare", "evolution", "all"):
        keys.add(f"view.{view}")
    assert keys - set(strings) == set()


def test_the_real_site_builds_in_both_languages_without_warnings(tmp_path, capsys):
    data = build_site.build(REAL_SITE, tmp_path / "out")
    out = tmp_path / "out"
    assert (out / "index.html").is_file() and (out / "it" / "index.html").is_file() and data["works"]
    assert (out / "it" / "opere" / "index.html").is_file() and (out / "works" / "index.html").is_file()
    assert 'data-page="works"' in (out / "works" / "index.html").read_text()
    assert capsys.readouterr().err == ""


# ---- which version is the preview ----------------------------------------------------------------------------------

def history_csv(best, mean):
    return f"generation,best_mse,mean_mse,elements\n0,0.1,0.2,10\n1,{best},{mean},20\n"


def work_with_versions(site, fitness, meta=None):
    """A work with one version per ``name: (best, mean)``; returns its folder."""
    folder = add_work(site, "w", final=False)
    for name, (best, mean) in fitness.items():
        add_version(folder, name)
        (folder / name / "history.csv").write_text(history_csv(best, mean))
    if meta is not None:
        (folder / "meta.json").write_text(json.dumps(meta))
    return folder


def test_the_preview_is_the_version_with_the_best_fitness(site, tmp_path):
    work_with_versions(site, {"disc": (0.003, 0.004), "polygon": (0.002, 0.003), "voronoi": (0.005, 0.006)})
    assert build_site.build(site, tmp_path / "out")["works"][0]["main"] == "polygon"


def test_the_average_fitness_counts_as_much_as_the_best_one(site, tmp_path):
    # polygon has the best image but a population that is far behind: disc wins on the mean of the two
    work_with_versions(site, {"disc": (0.0030, 0.0035), "polygon": (0.0025, 0.0200)})
    assert build_site.build(site, tmp_path / "out")["works"][0]["main"] == "disc"


def test_main_in_meta_json_overrides_the_fitness(site, tmp_path):
    work_with_versions(site, {"disc": (0.001, 0.001), "polygon": (0.009, 0.009)}, meta={"main": "polygon"})
    assert build_site.build(site, tmp_path / "out")["works"][0]["main"] == "polygon"


def test_a_main_that_does_not_exist_falls_back_to_the_fitness(site, tmp_path):
    work_with_versions(site, {"disc": (0.009, 0.009), "polygon": (0.001, 0.001)}, meta={"main": "nowhere"})
    assert build_site.build(site, tmp_path / "out")["works"][0]["main"] == "polygon"


def test_versions_without_a_history_come_after_those_with_one(site, tmp_path):
    folder = work_with_versions(site, {"voronoi": (0.5, 0.5)})
    add_version(folder, "disc")  # no history.csv
    assert build_site.build(site, tmp_path / "out")["works"][0]["main"] == "voronoi"


def test_the_featured_work_is_marked_in_the_data(site, tmp_path):
    add_work(site, "a")
    folder = add_work(site, "b")
    (folder / "meta.json").write_text(json.dumps({"featured": True}))
    works = {w["id"]: w for w in build_site.build(site, tmp_path / "out")["works"]}
    assert works["b"]["featured"] is True and works["a"]["featured"] is False


# ---- the credit of a work: artist, year, other titles --------------------------------------------------------------

def test_artist_year_and_other_titles_are_passed_and_localized(bilingual, tmp_path):
    folder = add_work(bilingual, "w")
    (folder / "meta.json").write_text(json.dumps({
        "title": "Donna con coppa", "artist": {"it": "Attribuito a Guido Reni", "en": "Attributed to Guido Reni"},
        "year": "1638–1639", "also_known_as": ["Artemisia", "Ritratto di donna"]}))
    out = tmp_path / "out"
    build_site.build(bilingual, out)
    work = {lang: json.loads((out / "data" / f"works.{lang}.json").read_text())["works"][0] for lang in ("it", "en")}
    assert (work["it"]["artist"], work["en"]["artist"]) == ("Attribuito a Guido Reni", "Attributed to Guido Reni")
    assert work["it"]["year"] == work["en"]["year"] == "1638–1639"
    assert work["en"]["also_known_as"] == ["Artemisia", "Ritratto di donna"]


def test_a_work_without_a_credit_has_empty_fields(site, tmp_path):
    add_work(site, "w")
    (work,) = build_site.build(site, tmp_path / "out")["works"]
    assert work["artist"] == "" and work["year"] == "" and work["also_known_as"] == []


def test_a_year_given_as_a_number_is_kept_as_text(site, tmp_path):
    folder = add_work(site, "w")
    (folder / "meta.json").write_text(json.dumps({"year": 1889}))
    assert build_site.build(site, tmp_path / "out")["works"][0]["year"] == "1889"


def test_location_and_credit_lines_are_localized_and_only_web_links_are_kept(bilingual, tmp_path):
    folder = add_work(bilingual, "w")
    (folder / "meta.json").write_text(json.dumps({
        "location": {"it": "Giardini Reali, Torino", "en": "Royal Gardens, Turin"},
        "credits": [{"label": {"it": "Foto su Unsplash", "en": "Photo on Unsplash"}, "url": "https://unsplash.com/photos/x"},
                    {"label": "Senza link"},
                    {"label": "Pericoloso", "url": "javascript:alert(1)"},
                    {"url": "https://example.org/senza-etichetta"},
                    "non un oggetto"]}))
    out = tmp_path / "out"
    build_site.build(bilingual, out)
    work = {lang: json.loads((out / "data" / f"works.{lang}.json").read_text())["works"][0] for lang in ("it", "en")}
    assert (work["it"]["location"], work["en"]["location"]) == ("Giardini Reali, Torino", "Royal Gardens, Turin")
    assert work["en"]["credits"] == [{"label": "Photo on Unsplash", "url": "https://unsplash.com/photos/x"},
                                      {"label": "Senza link"}, {"label": "Pericoloso"}]  # the javascript: link is dropped


def test_a_work_without_location_or_credits_has_them_empty(site, tmp_path):
    add_work(site, "w")
    (work,) = build_site.build(site, tmp_path / "out")["works"]
    assert work["location"] == "" and work["credits"] == []


# ---- the page with all the works ----------------------------------------------------------------------------------

WORKS_TEMPLATE = (
    '<html lang="{{lang}}" data-base="{{base}}" data-page="{{page}}"><head><title>{{page_title}}</title>{{{head_links}}}</head>'
    '<body>{{{partial.header}}}<a id="back" href="{{home_url}}">back</a>{{{lang_switch}}}<h1>{{works.page.title}}</h1></body></html>')
HOME_TEMPLATE = (
    '<html lang="{{lang}}" data-base="{{base}}" data-page="{{page}}"><head><title>{{page_title}}</title>{{{head_links}}}</head>'
    '<body>{{{partial.header}}}<a id="all" href="{{works_url}}">all</a>{{{lang_switch}}}</body></html>')


@pytest.fixture
def with_works_page(bilingual):
    (bilingual / "index.html").write_text(HOME_TEMPLATE)
    (bilingual / "works.html").write_text(WORKS_TEMPLATE)
    (bilingual / "partials").mkdir()
    (bilingual / "partials" / "header.html").write_text(
        '<nav><a id="brand" href="{{brand_url}}">{{site_title}}</a><a id="method" href="{{nav_method}}">{{nav.works}}</a></nav>')
    (bilingual / "i18n" / "it.json").write_text(json.dumps({**DICT_IT, "works.page.title": "Tutte le opere", "nav.works": "Opere"}))
    (bilingual / "i18n" / "en.json").write_text(json.dumps({**DICT_EN, "works.page.title": "All works", "nav.works": "Works"}))
    (bilingual / "site.json").write_text(json.dumps({"title": "gapoly", "url": "https://example.org/lab"}))
    return bilingual


def test_a_page_with_all_the_works_is_written_for_every_language(with_works_page, tmp_path):
    out = tmp_path / "out"
    build_site.build(with_works_page, out)
    pages = {"it": out / "it" / "opere" / "index.html", "en": out / "works" / "index.html"}
    assert all(page.is_file() for page in pages.values())
    assert "<h1>Tutte le opere</h1>" in pages["it"].read_text() and "<h1>All works</h1>" in pages["en"].read_text()
    assert 'data-base="../../"' in pages["it"].read_text() and 'data-base="../"' in pages["en"].read_text()
    assert 'data-page="works"' in pages["it"].read_text() and 'data-page="home"' in (out / "index.html").read_text()
    assert "gapoly — Tutte le opere" in pages["it"].read_text() and "gapoly — All works" in pages["en"].read_text()


def test_home_and_works_page_link_to_each_other_in_the_same_language(with_works_page, tmp_path):
    out = tmp_path / "out"
    build_site.build(with_works_page, out)
    assert 'id="all" href="opere/"' in (out / "it" / "index.html").read_text()
    assert 'id="all" href="works/"' in (out / "index.html").read_text()
    assert 'id="back" href="../"' in (out / "it" / "opere" / "index.html").read_text()
    assert 'id="back" href="../"' in (out / "works" / "index.html").read_text()  # works/ -> root


def test_the_header_of_the_works_page_leads_back_to_the_home(with_works_page, tmp_path):
    out = tmp_path / "out"
    build_site.build(with_works_page, out)
    home, works = (out / "index.html").read_text(), (out / "works" / "index.html").read_text()
    assert 'id="brand" href="#"' in home and 'id="method" href="#metodo"' in home
    assert 'id="brand" href="../"' in works and 'id="method" href="../#metodo"' in works


def test_the_language_switch_keeps_the_page(with_works_page, tmp_path):
    out = tmp_path / "out"
    build_site.build(with_works_page, out)
    italian_works = (out / "it" / "opere" / "index.html").read_text()
    english_works = (out / "works" / "index.html").read_text()
    assert 'href="./" hreflang="it"' in italian_works and 'href="../../works/" hreflang="en"' in italian_works
    assert 'href="../it/opere/" hreflang="it"' in english_works and 'href="./" hreflang="en"' in english_works
    assert 'href="it/" hreflang="it"' in (out / "index.html").read_text()  # the home switch leads to homes


def test_each_page_has_its_own_canonical_and_hreflang_links(with_works_page, tmp_path):
    out = tmp_path / "out"
    build_site.build(with_works_page, out)
    english_works = (out / "works" / "index.html").read_text()
    assert '<link rel="canonical" href="https://example.org/lab/works/">' in english_works
    assert '<link rel="alternate" hreflang="it" href="https://example.org/lab/it/opere/">' in english_works
    assert '<link rel="alternate" hreflang="x-default" href="https://example.org/lab/works/">' in english_works
    assert '<link rel="canonical" href="https://example.org/lab/">' in (out / "index.html").read_text()


def test_the_data_tells_how_many_works_the_home_shows(with_works_page, tmp_path):
    out = tmp_path / "out"
    build_site.build(with_works_page, out)
    for lang in ("it", "en"):
        assert json.loads((out / "data" / f"works.{lang}.json").read_text())["home_works"] == build_site.HOME_WORKS == 8


def test_without_a_works_template_there_is_only_the_home(bilingual, tmp_path):
    out = tmp_path / "out"
    build_site.build(bilingual, out)
    assert not (out / "it" / "opere").exists() and not (out / "works" / "index.html").exists()  # works/ holds the images


def test_the_history_of_an_old_run_with_a_polygons_column_is_read(site, tmp_path):
    folder = add_work(site, "w")
    (folder / "history.csv").write_text("generation,best_mse,mean_mse,polygons\n0,0.5,0.6,100\n1,0.25,0.3,133\n")
    (variant,) = build_site.build(site, tmp_path / "out")["works"][0]["variants"]
    assert variant["stats"] == {"generations": 1, "best_mse": 0.25, "mean_mse": 0.3, "elements": 133}
