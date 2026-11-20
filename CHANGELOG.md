# Changelog

## [0.3.0](https://github.com/carmelolg/GAPoly/compare/carmelolg-gapoly-v0.2.0...carmelolg-gapoly-v0.3.0) (2026-11-20)


### Features

* add mosaic, glyph and stroke shapes, rename repository to GAPoly ([53158de](https://github.com/carmelolg/GAPoly/commit/53158de74d67a35f354b5162756459a6896b2fca))
* **site:** gallery published on GitHub Pages with the lossless works optimizer ([74bcb01](https://github.com/carmelolg/GAPoly/commit/74bcb01978280f262f0251a2e2d399857193e010))

## [0.2.0](https://github.com/carmelolg/genetic-evolving-polygons/compare/carmelolg-gapoly-v0.1.0...carmelolg-gapoly-v0.2.0) (2026-11-18)


### ⚠ BREAKING CHANGES

* PolygonProblem, Kind, KINDS, COLS and the removed options no longer exist. Use ImageProblem(shape=...), Shape, SHAPES and --elements.

### Features

* add discs, selected with --shape disc ([f8630d4](https://github.com/carmelolg/genetic-evolving-polygons/commit/f8630d4cf82e37f16003ada00acbf8c8ae2142c1))
* add Voronoi points, selected with --shape voronoi ([77a99f8](https://github.com/carmelolg/genetic-evolving-polygons/commit/77a99f867e5c1454779e4b4d374eae9ac13eb85a))


### Code Refactoring

* shapes as a Strategy, drop obsolete options and dead code ([de74dde](https://github.com/carmelolg/genetic-evolving-polygons/commit/de74dde3f616f25affe8ac49aa5489cf6798f0cc))

## 0.1.0 (2026-11-18)


### Features

* gapoly, evolve semi-transparent polygons to approximate a photo ([4f8d1a6](https://github.com/carmelolg/genetic-evolving-polygons/commit/4f8d1a665a4d3768700b96dfef0688c2fa541c8b))
