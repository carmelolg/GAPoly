# Changelog

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
