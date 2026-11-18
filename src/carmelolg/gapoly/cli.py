from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from carmelolg.galib import GenerationStats, GeneticAlgorithm, PrintProgress
from PIL import Image, ImageOps

from .help_format import colorize, use_color
from .kinds import Kind
from .limits import Limits
from .problem import ImageProblem
from .viewer import LiveView, SizeSchedule, Snapshots


class Progress(PrintProgress):
    def __init__(self, kind: Kind, every: int = 1) -> None:
        super().__init__(every=every)
        self.kind = kind

    def format_line(self, stats: GenerationStats) -> str:
        return (f"gen {stats.generation:6d} | best mse {stats.best_fitness:.5f} | "
                f"mean {stats.mean_fitness:.5f} | {self.kind.plural} {len(stats.best_genome)}")


def load_target(path: Path, size: int) -> np.ndarray:
    """Load an image, scaled so its longest side is ``size`` pixels; float32 RGB in [0, 1]."""
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")  # honour the camera's rotation
    scale = size / max(image.size)
    image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


class _Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Keep the line breaks of description / epilog, print every default value, colour it on a terminal."""

    def format_help(self) -> str:
        text = super().format_help()
        return colorize(text) if use_color() else text


DESCRIPTION = """\
Ricrea una foto con poligoni semitrasparenti, usando un algoritmo genetico.

Ogni individuo è un'immagine fatta di poligoni (vertici e colore RGBA). A ogni generazione
sopravvivono gli individui più vicini alla foto (MSE più basso), si mutano e si incrociano.
"""

EPILOG = """\
Esempi:
  prova veloce:  gapoly foto.jpg --generations 300
  più fedele:    gapoly foto.jpg --generations 5000 --polygons 1500 --size 192 \\
                     --start-size 0.2 --max-size 0.04 --live

Tempi:
  Una generazione costa circa population x polygons x size². Parti da valori piccoli,
  leggi i secondi per generazione nel log (ogni 50) e poi scala.
  A parità di tempo, poca popolazione e molte generazioni rendono di più di una popolazione enorme.

Output in --out: final.png, evolution.gif, history.csv, snapshots/gen_*.png.
"""


METAVARS = {
    "generations": "N", "population": "N", "seed": "N", "polygons": "N", "initial_polygons": "N",
    "start_size": "FRAZIONE", "max_size": "FRAZIONE", "mutations_per_child": "N", "mutation_rate": "0-1",
    "crossover_rate": "0-1", "size": "PX", "output_width": "PX", "snapshot_every": "N", "history_every": "N", "out": "CARTELLA",
    "workers": "N", "live_every": "N",
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gapoly", description=DESCRIPTION, epilog=EPILOG,
                                formatter_class=_Formatter, add_help=False)
    p.add_argument("-h", "--help", action="help", help="mostra questo aiuto ed esci")
    p.add_argument("target", type=Path, help="foto da riprodurre (JPEG, PNG...); la rotazione EXIF viene applicata")

    run = p.add_argument_group("evoluzione")
    run.add_argument("--generations", type=int, default=3000,
                     help="generazioni da evolvere: più ne fai, più l'immagine si avvicina alla foto")
    run.add_argument("--population", type=int, default=40,
                     help="immagini candidate per generazione. Il costo cresce in modo lineare. "
                          "Non è il numero di poligoni (--polygons)")
    run.add_argument("--seed", type=int, default=None,
                     help="seme casuale: stesso seme, stessa evoluzione. Senza seme ogni corsa è diversa")

    shapes = p.add_argument_group("poligoni")
    shapes.add_argument("--polygons", type=int, default=400,
                        help="poligoni massimi per immagine. Più poligoni permettono più dettaglio "
                             "e rallentano ogni valutazione")
    shapes.add_argument("--initial-polygons", type=int, default=60,
                        help="poligoni di partenza di ogni individuo; salgono fino a --polygons")
    shapes.add_argument("--start-size", type=float, default=0.5,
                        help="lato massimo di un poligono all'inizio, come frazione del lato dell'immagine "
                             "(0.5 = metà immagine). Poligoni grandi coprono subito le zone di colore uniforme")
    shapes.add_argument("--max-size", type=float, default=0.12,
                        help="lato massimo finale. Il limite scende da --start-size a questo valore "
                             "nel primo 70 per cento delle generazioni, poi resta fisso. Più piccolo, più dettaglio")

    ops = p.add_argument_group("operatori genetici")
    ops.add_argument("--mutations-per-child", type=float, default=3.0,
                     help="poligoni modificati in media per figlio (vertice, spostamento, colore). "
                          "Troppi e quasi ogni figlio peggiora, pochi e l'evoluzione ristagna")
    ops.add_argument("--mutation-rate", type=float, default=0.2,
                     help="probabilità, per figlio, di aggiungere un poligono, di rimuoverne uno "
                          "e di cambiare l'ordine di sovrapposizione (ognuna per conto suo, da 0 a 1)")
    ops.add_argument("--crossover-rate", type=float, default=0.2,
                     help="probabilità che due genitori scambino una parte dei poligoni (da 0 a 1). "
                          "Con molti poligoni peggiora spesso i figli: tienila bassa")

    res = p.add_argument_group("risoluzione e output")
    res.add_argument("--size", type=int, default=96,
                     help="lato lungo in pixel a cui la foto viene ridotta per calcolare l'errore. "
                          "Decide velocità (cresce col quadrato) e dettaglio raggiungibile")
    res.add_argument("--output-width", type=int, default=512,
                     help="larghezza in pixel di final.png e degli snapshot. Non cambia l'evoluzione: "
                          "i poligoni si ridisegnano nitidi a qualsiasi misura")
    res.add_argument("--snapshot-every", type=int, default=100,
                     help="salva un PNG del migliore ogni N generazioni (1 = ogni generazione)")
    res.add_argument("--history-every", type=int, default=1000,
                     help="conserva in memoria il genoma migliore di una generazione ogni N, e di tutte le altre "
                          "solo l'errore. Evita che una corsa lunga con molti poligoni riempia la RAM "
                          "(history.csv e risultato finale non cambiano)")
    res.add_argument("--out", type=Path, default=None,
                     help="cartella di output. Se manca: out/<nome della foto>")

    perf = p.add_argument_group("prestazioni e finestra live")
    perf.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                      help="worker che valutano gli individui in parallelo (vedi --parallel). 1 = nessun parallelismo; "
                           "il default è il numero di CPU")
    perf.add_argument("--parallel", choices=["thread", "process"], default="thread",
                      help="tipo di worker. thread: il rendering Numba rilascia il GIL, nessuna copia dei dati. "
                           "process: processi separati (più memoria, utile solo se la fitness non rilascia il GIL)")
    perf.add_argument("--live", action="store_true",
                      help="apre una finestra con foto, miglior immagine e curva dell'errore; "
                           "i pannelli seguono l'orientamento della foto")
    perf.add_argument("--live-every", type=int, default=10,
                      help="aggiorna la finestra ogni N generazioni (richiede --live). Valori bassi la rallentano")
    for action in p._actions:  # short value placeholders read better than GENERATIONS, MAX_SIZE...
        action.metavar = METAVARS.get(action.dest, action.metavar)
    args = p.parse_args(argv)

    out = args.out or Path("out") / args.target.stem
    out.mkdir(parents=True, exist_ok=True)
    problem = ImageProblem(load_target(args.target, args.size), Limits(max_elements=args.polygons, max_size=max(args.start_size, args.max_size)),
                              args.initial_polygons, args.mutations_per_child)
    callbacks = [Progress(problem.kind, every=50), Snapshots(problem, out, args.snapshot_every, args.output_width),
                 SizeSchedule(problem, args.max_size, args.generations)]
    if args.live:
        callbacks.append(LiveView(problem, args.live_every))
    result = GeneticAlgorithm(problem, population_size=args.population, max_generations=args.generations,
                              mutation_rate=args.mutation_rate, crossover_rate=args.crossover_rate, callbacks=callbacks, seed=args.seed, diversity=None,
                              history_genome_every=args.history_every,
                              workers=args.workers if args.workers > 1 else None, parallel=args.parallel).run()
    print(f"best mse {result.best.fitness:.5f} after {result.generations} generations -> {out}/")
    return 0
