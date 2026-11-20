// Gallery: reads data/works.json (written by tools/build_site.py) and draws the page.
// A work is one photo with one or more versions (one per shape or per run). Everything that comes from the data is
// put in the page with textContent / setAttribute, never as HTML.
(() => {
  "use strict";

  // One small drawing per technique, for the "Metodo" section (the texts come from the dictionary).
  const SHAPE_ICONS = {
    polygon: "M6 32 18 8l16 20zM14 34l20-22 2 22z",
    disc: "M15 20a9 9 0 1 0 0 .01M27 26a8 8 0 1 0 0 .01M24 12a5 5 0 1 0 0 .01",
    voronoi: "M4 14 14 4M14 4l6 12M20 16 4 26M20 16l16-6M20 16l4 20M24 36 4 30M24 36l12-6",
    mosaic: "M12 6l6 3.5v7L12 20l-6-3.5v-7zM24 6l6 3.5v7L24 20l-6-3.5v-7zM18 17l6 3.5v7L18 31l-6-3.5v-7zM30 17l6 3.5v7L30 31l-6-3.5v-7z",
    glyph: "M6 32 13 8l7 24M8.5 24h9M24 32V16h6a4 4 0 0 1 0 8h-6M30 24a4 4 0 0 1 0 8",
    stroke: "M6 30C14 22 20 14 34 8M8 18c8 0 14 4 22 12M14 34c6-4 14-6 22-6",
  };
  const VIEWS = ["compare", "evolution", "all"];
  const SVG_NS = "http://www.w3.org/2000/svg";
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const PAGE = document.documentElement.dataset.page || "home"; // "home" shows the first works, "works" all of them

  // Language of this page: strings and locale come with the data, paths are relative to the page's folder.
  const BASE = document.documentElement.dataset.base || "";
  const u = (path) => (path ? BASE + path : path);
  let STR = {};
  let LOCALE = document.documentElement.lang || "it";
  const t = (key, vars) => {
    let text = STR[key] ?? key;
    if (vars) for (const [name, value] of Object.entries(vars)) text = text.replaceAll(`{${name}}`, value);
    return text;
  };
  const plural = (key, n, vars = {}) => t(`${key}_${new Intl.PluralRules(LOCALE).select(n) === "one" ? "one" : "other"}`, { n, ...vars });

  const $ = (id) => document.getElementById(id);
  const el = (tag, props = {}, children = []) => {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (value == null || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? "" : value);
    }
    for (const child of [].concat(children)) if (child) node.append(child);
    return node;
  };
  const svgEl = (tag, attrs = {}, children = []) => {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    for (const child of [].concat(children)) if (child) node.append(child);
    return node;
  };
  const icon = (d, extra = {}) => svgEl("svg", { viewBox: "0 0 24 24", "aria-hidden": "true", ...extra }, svgEl("path", { d }));
  const ICONS = {
    close: "M6 6l12 12M18 6 6 18", prev: "M15 6l-6 6 6 6", next: "M9 6l6 6-6 6",
    link: "M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1",
    expand: "M14 4h6v6M10 20H4v-6M20 4l-7 7M4 20l7-7", arrow: "M7 17 17 7M8 7h9v9", knob: "M9 7l-5 5 5 5M15 7l5 5-5 5",
  };

  const shapeLabel = (shape) => STR[`shape.${shape}`] ?? shape;
  const paramLabel = (key) => STR[`param.${key}`] ?? key;
  const number = (n) => new Intl.NumberFormat(LOCALE).format(n);
  const compact = (n) => new Intl.NumberFormat(LOCALE, { notation: n >= 10000 ? "compact" : "standard", maximumFractionDigits: 1 }).format(n);
  const toDate = (iso) => new Date(iso + "T12:00:00");
  const formatDate = (iso) => iso ? new Intl.DateTimeFormat(LOCALE, { year: "numeric", month: "long", day: "numeric" }).format(toDate(iso)) : "";
  const shortDate = (iso) => iso ? new Intl.DateTimeFormat(LOCALE, { year: "numeric", month: "short" }).format(toDate(iso)) : "";
  const ratio = (v) => `--w:${v.width};--h:${v.height}`; // the box keeps the picture's shape before it loads
  const credit = (work) => [work.artist, work.year].filter(Boolean).join(", "); // "Guido Reni, 1638–1639"
  const mainOf = (work) => work.variants.find((v) => v.id === work.main) || work.variants[0];
  const versionsText = (n) => plural("versions.n", n);
  const img = (props) => {
    const node = el("img", { decoding: "async", ...props, class: ["fade", props.class].filter(Boolean).join(" ") });
    const done = () => node.classList.add("is-loaded");
    if (node.complete && node.naturalWidth) done(); else { node.addEventListener("load", done); node.addEventListener("error", done); }
    return node;
  };

  let data = { site: {}, works: [] };
  let filter = null; // { kind: "technique" | "shape", value } or null for all

  // ----- before/after ------------------------------------------------------------------------------------------

  function compare(work, variant, { start = 50, hint = false } = {}) {
    const box = el("div", { class: "compare", style: `${ratio(variant)};--pos:${start}%` });
    box.append(img({ class: "after", src: u(variant.image), alt: t("alt.result", { title: work.title, label: variant.label }), width: variant.width, height: variant.height }));
    if (!work.original) return box;
    const slider = el("input", {
      type: "range", min: "0", max: "100", step: "0.5", value: String(start),
      "aria-label": t("compare.slider"),
    });
    const set = (value) => {
      box.style.setProperty("--pos", value + "%");
      box.dataset.side = value > 85 ? "left" : value < 15 ? "right" : "";
    };
    slider.addEventListener("input", () => { set(slider.value); box.classList.add("touched"); });
    slider.addEventListener("pointerdown", () => box.classList.add("is-dragging", "touched"));
    window.addEventListener("pointerup", () => box.classList.remove("is-dragging"));
    box.append(
      img({ class: "before", src: u(work.original), alt: t("alt.original", { title: work.title }), width: variant.width, height: variant.height }),
      el("span", { class: "rule" }, el("span", { class: "knob" }, icon(ICONS.knob))),
      el("span", { class: "tag l", text: t("compare.original") }),
      el("span", { class: "tag r", text: variant.label }),
      hint && el("span", { class: "hint", text: t("compare.hint") }),
      slider,
    );
    box.slider = slider;
    box.setPos = (v) => { slider.value = v; set(v); };
    return box;
  }

  // A short sweep of the divider, once, so it is obvious that the picture can be dragged.
  function sweep(box) {
    if (reduceMotion || !box.setPos) return;
    const keys = [[0, 50], [700, 22], [1500, 78], [2200, 50]];
    let t0 = null;
    const ease = (t) => t < .5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
    const step = (now) => {
      if (box.classList.contains("touched")) return;
      t0 ??= now;
      const t = now - t0;
      const i = keys.findIndex(([at]) => at > t);
      if (i === -1) { box.setPos(50); return; }
      const [a, va] = keys[i - 1], [b, vb] = keys[i];
      box.setPos((va + (vb - va) * ease((t - a) / (b - a))).toFixed(1));
      requestAnimationFrame(step);
    };
    const io = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) { io.disconnect(); setTimeout(() => requestAnimationFrame(step), 500); }
    }, { threshold: .6 });
    io.observe(box);
  }

  // ----- chart -------------------------------------------------------------------------------------------------

  let chartId = 0;
  function chart(values, { generations, W = 360, H = 150 } = {}) {
    if (!values || values.length < 2) return null;
    const L = 52, R = 6, T = 10, B = 22;
    const max = Math.max(...values), min = Math.min(...values);
    const span = max - min || 1;
    const x = (i) => L + (i * (W - L - R)) / (values.length - 1);
    const y = (v) => T + (1 - (v - min) / span) * (H - T - B);
    const genAt = (i) => generations ? Math.round((i * generations) / (values.length - 1)) : i;
    const id = `chart-fill-${++chartId}`;
    const pts = values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`);
    const cursor = svgEl("line", { class: "cursor", y1: T, y2: H - B, x1: -10, x2: -10 });
    const dot = svgEl("circle", { class: "cursor-dot", r: 4, cx: -10, cy: -10 });
    const svg = svgEl("svg", {
      viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": generations
        ? t("chart.aria", { max: max.toFixed(4), min: min.toFixed(4), gens: number(generations) })
        : t("chart.aria_plain", { max: max.toFixed(4), min: min.toFixed(4) }),
    }, [
      svgEl("defs", {}, svgEl("linearGradient", { id, x1: 0, y1: 0, x2: 0, y2: 1 }, [
        svgEl("stop", { offset: "0", "stop-color": "var(--accent)", "stop-opacity": ".28" }),
        svgEl("stop", { offset: "1", "stop-color": "var(--accent)", "stop-opacity": "0" }),
      ])),
      ...[max, (max + min) / 2, min].flatMap((v) => [
        svgEl("line", { class: "grid-line", x1: L, x2: W - R, y1: y(v), y2: y(v) }),
        svgEl("text", { x: L - 8, y: y(v) + 3.5, "text-anchor": "end" }, document.createTextNode(v.toFixed(4))),
      ]),
      svgEl("path", { d: `M${pts[0]} L${pts.join(" L")} L${x(values.length - 1)},${H - B} L${L},${H - B}Z`, fill: `url(#${id})` }),
      svgEl("polyline", { class: "line", points: pts.join(" ") }),
      svgEl("text", { x: L, y: H - 5 }, document.createTextNode(t("chart.gen", { n: 0 }))),
      svgEl("text", { x: W - R, y: H - 5, "text-anchor": "end" }, document.createTextNode(t("chart.gen", { n: generations ? number(generations) : values.length - 1 }))),
      cursor, dot,
    ]);
    const left = el("span"), right = el("span");
    const idle = () => {
      left.replaceChildren(t("chart.final") + " ", el("b", { text: min.toFixed(5) }));
      right.textContent = t("chart.drop", { pct: ((1 - min / max) * 100).toFixed(1) });
      wrap.classList.remove("is-active");
    };
    const move = (event) => {
      wrap.classList.add("is-active");
      const box = svg.getBoundingClientRect();
      const sx = ((event.clientX - box.left) / box.width) * W;
      const i = Math.max(0, Math.min(values.length - 1, Math.round(((sx - L) / (W - L - R)) * (values.length - 1))));
      cursor.setAttribute("x1", x(i)); cursor.setAttribute("x2", x(i));
      dot.setAttribute("cx", x(i)); dot.setAttribute("cy", y(values[i]));
      left.replaceChildren(t("chart.at") + " ", el("b", { text: number(genAt(i)) }));
      right.replaceChildren(t("chart.err") + " ", el("b", { text: values[i].toFixed(5) }));
    };
    const wrap = el("div", { class: "chart" });
    svg.addEventListener("pointermove", move);
    svg.addEventListener("pointerdown", move);
    svg.addEventListener("pointerleave", idle);
    idle();
    wrap.append(svg, el("div", { class: "chart-readout", "aria-hidden": "true" }, [left, right]));
    return wrap;
  }

  // ----- hero --------------------------------------------------------------------------------------------------

  function renderHero() {
    const hero = $("hero");
    if (!hero) return; // the page of all the works has no hero
    const work = data.works.find((w) => w.featured) || data.works[0];
    if (!work) { hero.hidden = true; return; }
    hero.hidden = false;
    let current = mainOf(work);
    const holder = el("div");
    const show = (variant, first = false) => {
      const pos = holder.firstChild?.slider?.value ?? 50;
      hero.style.setProperty("--w", variant.width);
      hero.style.setProperty("--h", variant.height);
      const box = compare(work, variant, { start: first ? 50 : pos, hint: first });
      if (!first) box.classList.add("touched");
      holder.replaceChildren(box);
      if (first) sweep(box);
    };
    show(current, true);
    const seg = work.variants.length > 1 && el("div", { class: "seg", role: "group", "aria-label": t("hero.technique_aria") },
      work.variants.map((v) => el("button", {
        type: "button", "aria-pressed": String(v.id === current.id), text: v.label,
        onclick: (e) => {
          current = v; show(v);
          seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === e.currentTarget)));
        },
      })));
    hero.replaceChildren(holder, el("div", { class: "hero-art-bar" }, [
      el("figcaption", {}, [
        el("span", { class: "t", text: work.title }),
        el("span", { class: "s", text: [t("hero.featured"), credit(work) || (work.date && shortDate(work.date)), versionsText(work.variants.length)].filter(Boolean).join(" · ") }),
      ]),
      seg,
      el("button", { type: "button", class: "icon-btn", "aria-label": t("hero.open", { title: work.title }), onclick: () => openWork(work.id, { variant: current.id }) }, icon(ICONS.expand)),
    ]));

    // a few honest numbers, straight from the data
    const versions = data.works.reduce((n, w) => n + w.variants.length, 0);
    const gens = data.works.reduce((n, w) => n + w.variants.reduce((m, v) => m + (v.stats?.generations || 0), 0), 0);
    const shapes = new Set(data.works.flatMap((w) => w.shapes)).size;
    const stats = [[data.works.length, plural("stat.works", data.works.length)], [versions, plural("stat.versions", versions)]];
    if (shapes) stats.push([shapes, plural("stat.techniques", shapes)]);
    if (gens) stats.push([compact(gens), t("stat.generations")]);
    $("hero-stats").replaceChildren(...stats.map(([v, k]) => el("div", {}, [el("dt", { text: k }), el("dd", { text: String(v) })])));
    $("hero-stats").hidden = false;
  }

  // ----- gallery -----------------------------------------------------------------------------------------------

  const matches = (work) => !filter || (filter.kind === "technique" ? work.technique === filter.value : work.shapes.includes(filter.value));

  function renderFilters() {
    const box = $("filters");
    if (!box) return; // the filters are on the page of all the works
    const options = [];
    const techniques = [...new Set(data.works.map((w) => w.technique).filter(Boolean))];
    const shapes = [...new Set(data.works.flatMap((w) => w.shapes))];
    const count = (o) => data.works.filter((w) => o.kind === "technique" ? w.technique === o.value : w.shapes.includes(o.value)).length;
    if (techniques.length > 1) techniques.forEach((t) => options.push({ kind: "technique", value: t, label: t }));
    if (shapes.length > 1) shapes.forEach((s) => options.push({ kind: "shape", value: s, label: shapeLabel(s) }));
    box.hidden = options.length === 0 || data.works.length < 2;
    if (box.hidden) return;
    const pick = (next) => { filter = next; renderFilters(); renderGrid(); };
    const chip = (label, n, pressed, onclick) => el("button", { type: "button", class: "chip", "aria-pressed": String(pressed), onclick }, [label, el("span", { class: "n", text: String(n) })]);
    box.replaceChildren(
      chip(t("filters.all"), data.works.length, !filter, () => pick(null)),
      ...options.map((o) => chip(o.label, count(o), !!filter && filter.kind === o.kind && filter.value === o.value, () => pick(o))),
    );
  }

  function renderGrid() {
    const matching = data.works.filter(matches);
    const shown = matching.slice(0, PAGE === "home" ? data.home_works || 8 : Infinity);
    $("empty").hidden = data.works.length > 0;
    $("count").textContent = !data.works.length ? ""
      : shown.length < matching.length ? t("count.of", { shown: shown.length, total: matching.length }) : plural("count.n", matching.length);
    const more = $("more");
    if (more) { // "see all", only when the home leaves some works out
      more.hidden = matching.length <= shown.length;
      $("see-all").textContent = t("works.see_all", { n: matching.length });
    }
    $("grid").replaceChildren(...shown.map((work) => {
      const cover = mainOf(work);
      const n = work.variants.length;
      return el("button", {
        type: "button", class: "card reveal" + (work.original ? " has-orig" : ""), "aria-label": t("card.aria", { title: work.title, versions: versionsText(n) }),
        onclick: () => openWork(work.id),
      }, [
        el("span", { class: "card-media", style: ratio(cover) }, [
          img({ class: "main", src: u(cover.thumb), alt: "", loading: "lazy", width: cover.width, height: cover.height }),
          work.original && el("img", { class: "orig", src: u(work.original), alt: "", loading: "lazy", width: cover.width, height: cover.height }),
          el("span", { class: "peek" }, [el("span", { class: "a", text: cover.label }), el("span", { class: "b", text: t("compare.original") })]),
          el("span", { class: "open", "aria-hidden": "true" }, icon(ICONS.arrow)),
        ]),
        el("span", { class: "card-body" }, [
          el("h3", { text: work.title }),
          credit(work) && el("span", { class: "card-credit", text: credit(work) }),
          el("span", { class: "card-meta" }, [
            n > 1 && el("span", { class: "swatches", "aria-hidden": "true" },
              work.variants.slice(0, 4).map((v) => { const i = el("i"); i.style.backgroundImage = `url("${encodeURI(u(v.thumb))}")`; return i; })),
            el("span", { text: [n > 1 ? versionsText(n) : cover.label, work.date && shortDate(work.date)].filter(Boolean).join(" · ") }),
          ]),
        ]),
      ]);
    }));
    observeReveal();
  }

  // ----- method ------------------------------------------------------------------------------------------------

  function renderMethod() {
    if (!$("shapes")) return; // only the home has the method section
    const present = new Set(data.works.flatMap((w) => w.shapes));
    $("shapes").replaceChildren(...Object.keys(SHAPE_ICONS).map((key) => el("li", { class: present.has(key) ? "on" : null }, [
      svgEl("svg", { viewBox: "0 0 40 40", "aria-hidden": "true" }, svgEl("path", { d: SHAPE_ICONS[key] })),
      el("div", {}, [
        el("strong", {}, [shapeLabel(key), present.has(key) && el("span", { class: "pill", text: t("shapes.present") })]),
        el("span", { text: t(`shape.${key}.text`) }),
      ]),
    ])));

    const work = data.works.find((w) => w.featured) || data.works[0];
    const variant = work && mainOf(work);
    const big = variant && chart(variant.history, { generations: variant.stats?.generations, ...(matchMedia("(max-width: 600px)").matches ? { W: 360, H: 190 } : { W: 640, H: 260 }) });
    $("curve").hidden = !big;
    if (big) {
      $("curve-chart").replaceChildren(big);
      $("curve-caption").textContent = t("curve.caption", { title: work.title, label: variant.label });
    }
  }

  // ----- viewer ------------------------------------------------------------------------------------------------

  const dialog = () => $("detail");
  const state = { work: null, variant: null, view: "compare", pos: 50 };

  function visibleWorks() { const list = data.works.filter(matches); return list.some((w) => w.id === state.work?.id) ? list : data.works; }

  function renderViewer({ focus } = {}) {
    const { work } = state;
    const variant = work.variants.find((v) => v.id === state.variant) || mainOf(work);
    const list = visibleWorks();
    const index = list.findIndex((w) => w.id === work.id);
    const go = (d) => { const next = list[index + d]; if (next) openWork(next.id); };
    const setVariant = (id, view = state.view === "all" ? "compare" : state.view) => {
      const box = $("detail-content").querySelector(".compare");
      if (box?.slider) state.pos = Number(box.slider.value);
      state.variant = id; state.view = view; renderViewer();
    };
    const hasGif = !!variant.gif;
    const views = VIEWS.filter((key) => (key !== "evolution" || hasGif) && (key !== "all" || work.variants.length > 1 || work.original));
    if (!views.includes(state.view)) state.view = "compare";

    let canvas;
    if (state.view === "evolution") {
      canvas = el("div", {}, [
        el("div", { class: "evolution", style: ratio(variant) },
          img({ src: u(variant.gif), alt: t("alt.evolution", { title: work.title, label: variant.label }), width: variant.width, height: variant.height })),
        el("p", { class: "evo-note", text: variant.stats ? t("viewer.evo_note", { n: number(variant.stats.generations) }) : t("viewer.evo_plain") }),
      ]);
    } else if (state.view === "all") {
      canvas = allVersions(work, (id) => setVariant(id, "compare"));
    } else {
      canvas = compare(work, variant, { start: state.pos });
      canvas.classList.add("touched");
    }

    const bar = el("div", { class: "v-bar" }, [
      el("button", { type: "button", class: "icon-btn", id: "detail-close", "aria-label": t("viewer.close"), onclick: () => dialog().close() }, icon(ICONS.close)),
      el("div", { class: "crumb" }, [el("b", { text: work.title }), list.length > 1 && el("span", { class: "pos", text: `${index + 1} / ${list.length}` })]),
      el("button", { type: "button", class: "icon-btn", "aria-label": t("viewer.copy"), onclick: share }, icon(ICONS.link)),
      list.length > 1 && el("button", { type: "button", class: "icon-btn", "aria-label": t("viewer.prev"), disabled: index <= 0, onclick: () => go(-1) }, icon(ICONS.prev)),
      list.length > 1 && el("button", { type: "button", class: "icon-btn", "aria-label": t("viewer.next"), disabled: index >= list.length - 1, onclick: () => go(1) }, icon(ICONS.next)),
    ]);

    const viewSeg = views.length > 1 && el("div", { class: "v-views" }, el("div", { class: "seg", role: "group", "aria-label": t("viewer.views_aria") },
      views.map((key) => el("button", { type: "button", "aria-pressed": String(state.view === key), text: t(`view.${key}`), onclick: () => { state.view = key; renderViewer(); } }))));

    const side = el("aside", { class: "v-side" }, [
      el("div", {}, [
        badges(work),
        el("h2", { id: "detail-title", text: work.title }),
        credit(work) && el("p", { class: "credit", text: credit(work) }),
        work.location && el("p", { class: "desc loc", text: work.location }),
        work.also_known_as.length > 0 && el("p", { class: "desc aka", text: t("viewer.aka", { names: work.also_known_as.join(", ") }) }),
        work.date && el("p", { class: "desc", text: t("viewer.made", { date: formatDate(work.date) }) }),
        work.description && el("p", { class: "desc", text: work.description }),
      ]),
      work.variants.length > 1 && el("div", {}, [
        el("p", { class: "v-label", text: t("viewer.versions") }),
        el("div", { class: "vers", role: "group", "aria-label": t("viewer.versions") }, work.variants.map((v) => el("button", {
          type: "button", class: "ver", "aria-pressed": String(state.view !== "all" && v.id === variant.id), onclick: () => setVariant(v.id),
        }, [el("img", { src: u(v.thumb), alt: "", loading: "lazy" }), el("span", { text: v.label })]))),
      ]),
      state.view !== "all" && facts(variant),
      state.view !== "all" && variant.history?.length > 1 && el("div", {}, [
        el("p", { class: "v-label", text: t("viewer.error_per_gen") }),
        chart(variant.history, { generations: variant.stats?.generations }),
      ]),
      state.view === "all" && el("p", { class: "muted", text: t("viewer.all_hint") }),
      work.credits.length > 0 && el("ul", { class: "credits" }, work.credits.map((c) =>
        el("li", {}, c.url ? el("a", { href: c.url, rel: "noopener", target: "_blank", text: c.label }) : c.label))),
      work.variants.length > 1 && el("p", { class: "kbd-hint" }, [el("kbd", { text: "←" }), " ", el("kbd", { text: "→" }), ` ${t("viewer.kbd_switch")} · `, el("kbd", { text: "Esc" }), ` ${t("viewer.kbd_close")}`]),
    ]);

    const quick = work.variants.length > 1 && state.view !== "all" && el("div", { class: "v-quick" }, el("div", { class: "seg", role: "group", "aria-label": t("viewer.version_aria") },
      work.variants.map((v) => el("button", { type: "button", "aria-pressed": String(v.id === variant.id), text: v.label, onclick: () => setVariant(v.id) }))));

    $("detail-content").replaceChildren(bar, el("div", { class: "v-stage" }, [viewSeg, el("div", { class: "v-canvas" }, canvas), quick]), side);
    if (focus) $("detail-close").focus();
  }

  function allVersions(work, onPick) {
    const base = mainOf(work);
    const items = [];
    if (work.original) items.push(el("figure", { class: "version", style: ratio(base) }, [
      el("img", { src: u(work.original), alt: t("alt.original", { title: work.title }), loading: "lazy", width: base.width, height: base.height }),
      el("figcaption", { class: "caption", text: t("compare.original") }),
    ]));
    for (const v of work.variants) items.push(el("button", { type: "button", class: "version", style: ratio(v), "aria-label": t("viewer.compare_with", { label: v.label }), onclick: () => onPick(v.id) }, [
      el("img", { src: u(v.thumb), alt: `${work.title}, ${v.label}`, loading: "lazy", width: v.width, height: v.height }),
      el("span", { class: "caption" }, [v.label, v.stats && el("small", { text: t("viewer.error_small", { value: v.stats.best_mse.toFixed(4) }) })]),
    ]));
    return el("div", { class: "all-grid" }, items);
  }

  function badges(work) {
    return el("div", { class: "badges" }, [
      work.technique && el("span", { class: "badge", text: work.technique }),
      ...work.tags.map((tag) => el("span", { class: "badge", text: "#" + tag })),
    ]);
  }

  function facts(variant) {
    const parts = [];
    if (variant.stats) {
      parts.push(el("dl", { class: "stats" }, [
        [t("fact.generations"), compact(variant.stats.generations)], [t("fact.elements"), number(variant.stats.elements)], [t("fact.error"), variant.stats.best_mse.toFixed(4)],
      ].map(([k, v]) => el("div", {}, [el("dt", { text: k }), el("dd", { text: v })]))));
    }
    const rows = [];
    if (variant.shape) rows.push([t("fact.shape"), shapeLabel(variant.shape)]);
    for (const [key, value] of Object.entries(variant.params || {})) {
      if (variant.stats && ["generations", "elements"].includes(key)) continue;
      rows.push([paramLabel(key), typeof value === "number" ? number(value) : String(value)]);
    }
    if (rows.length) {
      const list = el("dl", { class: "params" });
      for (const [name, value] of rows) list.append(el("dt", { text: name }), el("dd", { text: value }));
      parts.push(list);
    }
    return parts.length ? el("div", {}, [el("p", { class: "v-label", text: t("viewer.data") }), ...parts]) : null;
  }

  async function share() {
    const url = location.href;
    try {
      if (navigator.share && matchMedia("(pointer: coarse)").matches) { await navigator.share({ title: state.work.title, url }); return; }
      await navigator.clipboard.writeText(url);
      toast(t("toast.copied"));
    } catch (e) { /* the user cancelled, or the clipboard is not available */ }
  }
  function toast(text) {
    const node = el("div", { class: "toast", role: "status", text });
    dialog().append(node);
    setTimeout(() => node.remove(), 2300);
  }

  function openWork(id, { push = true, variant } = {}) {
    const work = data.works.find((w) => w.id === id);
    if (!work) return;
    const opening = !dialog().open;
    state.work = work;
    state.variant = variant || mainOf(work).id;
    if (opening) { state.view = "compare"; state.pos = 50; }
    renderViewer({ focus: opening });
    if (opening) { dialog().showModal(); document.body.classList.add("viewer-open"); }
    dialog().scrollTop = 0;
    if (push) history.replaceState(null, "", "#" + encodeURIComponent(id));
  }

  function closeWork() {
    document.body.classList.remove("viewer-open");
    if (location.hash && data.works.some((w) => w.id === decodeURIComponent(location.hash.slice(1)))) {
      history.replaceState(null, "", location.pathname + location.search);
    }
  }

  function openFromHash() {
    const id = decodeURIComponent(location.hash.slice(1));
    if (data.works.some((w) => w.id === id)) openWork(id, { push: false });
    else if (dialog().open) dialog().close();
  }

  function onViewerKey(event) {
    if (!state.work || event.altKey || event.metaKey || event.ctrlKey) return;
    if (event.target.matches?.("input, textarea")) return; // the slider uses the arrows itself
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    const vs = state.work.variants;
    if (vs.length < 2) return;
    event.preventDefault();
    const i = vs.findIndex((v) => v.id === state.variant);
    const box = $("detail-content").querySelector(".compare");
    if (box?.slider) state.pos = Number(box.slider.value);
    state.variant = vs[(i + (event.key === "ArrowRight" ? 1 : -1) + vs.length) % vs.length].id;
    if (state.view === "all") state.view = "compare";
    renderViewer();
  }

  // ----- page chrome -------------------------------------------------------------------------------------------

  function renderSite(site) {
    const title = site.title || "GAPoly";
    document.title = site.tagline ? `${title} — ${site.tagline}` : title;
    $("site-title").textContent = title;
    const set = (id, text) => { const node = $(id); if (node) node.textContent = text; }; // not every page has every block
    set("site-description", site.description || "");
    if (site.tagline) set("intro-title", site.tagline);
    for (const sel of ['meta[name="description"]', 'meta[property="og:description"]']) {
      const meta = document.querySelector(sel);
      if (meta && site.description) meta.setAttribute("content", site.description);
    }
    if (site.logo) { const logo = $("logo"); logo.src = u(site.logo); logo.hidden = false; $("brand-mark").remove(); }
    if (site.favicon) document.head.append(el("link", { rel: "icon", href: u(site.favicon) }));
    if (site.repo_url) { $("repo-link").href = site.repo_url; $("repo-link").hidden = false; }
    $("about")?.replaceChildren(...(site.about || []).map((text) => el("p", { text })));
    $("links")?.replaceChildren(...(site.links || []).map((l) => {
      let host = "";
      try { host = new URL(l.url).host.replace(/^www\./, ""); } catch (e) { /* relative link */ }
      return el("li", {}, el("a", { href: l.url, rel: "noopener" }, [el("span", {}, [l.label, host && el("small", { text: host })]), icon(ICONS.arrow)]));
    }));
    $("foot-text").textContent = site.author ? `© ${new Date().getFullYear()} ${site.author} · ${title}` : title;
    // The licence comes from site.json: {"license": {"name": "...", "url": "..."}}; without it nothing is shown.
    const license = site.license;
    if (license && license.name) {
      const link = $("foot-license-link");
      link.textContent = t("foot.license", { name: license.name });
      if (license.url) link.href = license.url; else link.removeAttribute("rel");
      $("foot-license").hidden = false;
    }
  }

  function setupMenu() {
    const button = $("menu-toggle"), panel = $("top-extra");
    if (!button || !panel) return;
    const set = (open) => { panel.classList.toggle("is-open", open); button.setAttribute("aria-expanded", String(open)); };
    button.addEventListener("click", () => set(!panel.classList.contains("is-open")));
    document.addEventListener("click", (e) => { if (!panel.contains(e.target) && !button.contains(e.target)) set(false); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && panel.classList.contains("is-open")) { set(false); button.focus(); } });
  }

  function setupTheme() {
    const button = $("theme-toggle");
    const media = matchMedia("(prefers-color-scheme: light)");
    const current = () => document.documentElement.dataset.theme || (media.matches ? "light" : "dark");
    const label = () => button.setAttribute("aria-label", current() === "dark" ? t("theme.to_light") : t("theme.to_dark"));
    button.addEventListener("click", () => {
      const next = current() === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem("gapoly-theme", next); } catch (e) { /* private mode */ }
      document.querySelector('meta[name="theme-color"]')?.setAttribute("content", next === "dark" ? "#0d0d0c" : "#f3f1ec");
      label();
    });
    label();
  }

  function setupScroll() {
    const top = $("top");
    const onScroll = () => top.classList.toggle("is-scrolled", scrollY > 8);
    addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    const links = [...document.querySelectorAll(".nav a")];
    const inPage = links.filter((a) => a.getAttribute("href").startsWith("#")); // the others lead to another page
    const sections = [document.querySelector(".hero"), ...inPage.map((a) => document.querySelector(a.getAttribute("href")))].filter(Boolean);
    const spy = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        for (const a of links) {
          if (a.getAttribute("href") === "#" + entry.target.id) a.setAttribute("aria-current", "true");
          else a.removeAttribute("aria-current");
        }
      }
    }, { rootMargin: "-45% 0px -50% 0px" });
    sections.forEach((s) => spy.observe(s));
  }

  let revealer;
  function observeReveal() {
    document.querySelectorAll(".section-head, .steps, .curve, .shapes-block, .info-grid").forEach((n) => n.classList.add("reveal"));
    if (reduceMotion || !("IntersectionObserver" in window)) { document.querySelectorAll(".reveal").forEach((n) => n.classList.add("in")); return; }
    revealer ??= new IntersectionObserver((entries) => {
      for (const entry of entries) if (entry.isIntersecting) { entry.target.classList.add("in"); revealer.unobserve(entry.target); }
    }, { rootMargin: "0px 0px -8% 0px" });
    document.querySelectorAll(".reveal:not(.in)").forEach((n) => revealer.observe(n));
  }

  async function init() {
    setupScroll();
    dialog().addEventListener("close", () => { $("detail-content").replaceChildren(); state.work = null; closeWork(); });
    document.addEventListener("keydown", (e) => { if (dialog().open) onViewerKey(e); });
    addEventListener("hashchange", openFromHash);
    try {
      const response = await fetch(u(document.documentElement.dataset.works || "data/works.it.json"), { cache: "no-cache" });
      if (!response.ok) throw new Error(response.status);
      data = await response.json();
    } catch (error) {
      console.error("Cannot load the works data", error);
    }
    STR = data.strings || {};
    LOCALE = data.locale || LOCALE;
    setupTheme();
    setupMenu();
    data.works ||= [];
    for (const w of data.works) { w.shapes ||= []; w.tags ||= []; w.variants ||= []; w.also_known_as ||= []; w.credits ||= []; }
    data.works = data.works.filter((w) => w.variants.length);
    renderSite(data.site || {});
    renderHero();
    renderFilters();
    renderGrid();
    renderMethod();
    observeReveal();
    openFromHash();
  }

  init();
})();
