# Sito galleria (note per me)

Queste note non vanno nel README del progetto: servono solo a gestire il sito. Questa cartella non viene pubblicata
(il build copia solo `index.html`, `css/`, `js/`, il logo e le opere elaborate).

## Impostazione una tantum
Repository su GitHub: Settings, Pages, Source: **GitHub Actions**. Poi ogni push su `main` che tocca `site/**` ripubblica
il sito (workflow `.github/workflows/pages.yml`); si può anche avviare a mano dalla scheda Actions.

## Aggiungere un'opera
1. Crea `site/assets/works/AAAA-MM-GG_nome-opera/` (la data ordina, il nome diventa il titolo).
2. Aggiungi `original.jpg` (la foto di partenza, abilita il confronto prima/dopo) e, se vuoi, un `meta.json`
   (modello in `assets/works/_template/`).
3. Una sottocartella per **versione** (per forma o per esecuzione), con dentro quello che gapoly scrive in `--out`,
   senza rinominare i file: `final.png`, `evolution.gif`, `history.csv`. Per esempio `disc/`, `mosaic/`, `polygon-hq/`.
   Nel sito compaiono come pulsanti e nella vista "Tutte le versioni". Una versione sola: file direttamente nella cartella.
4. Commit e push: il sito si rigenera da solo. Dettagli in `assets/works/README.md`.

## Logo
In `assets/logo/` come `logo.svg` (o `.png`, `.webp`) e, opzionale, `favicon.png`. Vedi `assets/logo/README.md`.

## Testi del sito
In `site.json`: titolo, sottotitolo, descrizione, autore (con `author_url` il nome diventa un link nel piè di pagina), link, paragrafi "Info", `url` del sito (serve ai link
canonical/hreflang) e, se vuoi, la licenza nel piè di pagina
(`"license": {"name": "Apache-2.0", "url": "https://..."}`; senza il campo non compare). Accanto si può mostrare
la licenza delle opere con `"works_license": {"name": "CC BY-NC-SA 4.0", "url": "https://..."}` (stesso formato).

## Due lingue (it / en)
- La radice del sito (`/`) è in italiano, l'inglese sta in `/en/`; nell'intestazione c'è il selettore IT | EN.
- Tutti i testi della pagina e dello script sono in `i18n/it.json` e `i18n/en.json` (stesse chiavi: lo verifica un test).
  `index.html` è un modello: il build ne scrive una pagina per lingua riempiendo i segnaposto `{{chiave}}`
  (testo) e `{{{chiave}}}` (HTML).
- Titoli, descrizioni, tag, etichette e testi di `site.json` possono essere una stringa sola (uguale per tutte le
  lingue) oppure `{"it": "...", "en": "..."}`. Se manca una lingua si usa l'italiano.
- Per una lingua in più: un file `i18n/<lingua>.json` e la lingua in `LANGUAGES` di `tools/build_site.py`.
- Se togli `i18n/en.json` il sito torna monolingua, senza selettore.

## Pagina con tutte le opere
- La home mostra le prime 8 opere (le più recenti: `HOME_WORKS` in `tools/build_site.py`); se ce ne sono di più, sotto la
  griglia compare il pulsante "Vedi tutte le opere (N)" che porta alla pagina dedicata.
- La pagina dedicata è `/opere/` in italiano e `/en/works/` in inglese: ha tutte le opere e i filtri per tecnica e forma
  (in home i filtri non ci sono). Il visualizzatore e i link `#nome-opera` funzionano su entrambe.
- Le pagine sono modelli in `site/`: `index.html` (home), `works.html` (tutte le opere) e le parti comuni in
  `site/partials/` (`head`, `header`, `footer`, `viewer`). Il selettore di lingua porta alla stessa pagina nell'altra lingua.

## Copertina di ogni opera
- **Opera in evidenza (hero della home):** quella con `"featured": true` nel suo `meta.json` (ora Artemisia). Senza
  nessuna, la più recente.
- **Anteprima di un'opera** (scheda in galleria e apertura del visualizzatore): in automatico la versione con la
  fitness migliore, cioè il valore più basso della media tra l'errore finale dell'immagine migliore (`best_mse`) e
  l'errore medio di tutta la popolazione (`mean_mse`) all'ultima generazione, letti da `history.csv`. Le versioni senza
  `history.csv` vengono dopo. `"main": "polygon"` nel `meta.json` dell'opera forza una versione.

## Alleggerire il repository
`tools/optimize_works.py` riduce le cartelle delle opere senza cambiare nulla di ciò che il sito mostra: tiene in
`history.csv` solo le righe che servono (grafico e statistiche), converte `final.png` in WebP senza perdita (stessi pixel,
verificato), riduce la foto a 1600 px di larghezza (JPEG qualità 95) e porta la GIF alla versione che il sito pubblica
(poche generazioni, 128 colori) scrivendola come WebP animato senza perdita (`evolution.webp`: stessi fotogrammi, stessi
tempi; le GIF con pochi fotogrammi, dove il WebP non conviene, restano GIF). Il build accetta sia `evolution.gif` sia
`evolution.webp`.
```bash
python tools/optimize_works.py --dry-run   # solo il rapporto
python tools/optimize_works.py             # applica (i file originali di gapoly restano dove li hai: tienili)
```
È idempotente. Si lancia dopo aver copiato una nuova opera, prima del commit: quando mi chiedi di aggiungerne una
lo faccio io (copio i file nella struttura, scrivo il `meta.json`, lancio lo strumento e controllo il build).

## Anteprima in locale
```bash
python tools/build_site.py --serve     # builda in _site/ e serve su http://127.0.0.1:8000
uv run tools/build_site.py --serve     # uguale, senza ambiente preparato
```
Per vedere le modifiche rilancia il comando. Aprendo `index.html` con doppio clic non carica i dati.

## Cose da ricordare
- **Release:** release-please ignora i commit che toccano solo `site/`, `tools/` e `pages.yml` (`exclude-paths` in
  `release-please-config.json`). Tieni separati i commit del sito da quelli del codice.
- **Peso:** il sito pubblica versioni leggere (immagini ridotte, GIF a 640 px); nel repository restano i file pieni.
  GitHub avvisa sopra i 50 MB per file e rifiuta sopra i 100 MB: una `evolution.gif` di gapoly può pesare 25-40 MB,
  riducila prima di copiarla se vuoi un repository leggero.
- **Altre tecniche** (automi cellulari, ecc.): stessa cartella, bastano `final.png` o `evolution.gif` e un `meta.json`
  con `"technique": "Automa cellulare"`; la tecnica diventa un filtro nella galleria.
- **Build:** `tools/build_site.py` (Pillow), con i test in `tests/test_site_build.py`.
- **Aspetto:** tema scuro/chiaro che segue il sistema, con interruttore (la scelta resta in `localStorage["gapoly-theme"]`).
  I font (Instrument Serif, Inter, JetBrains Mono) arrivano da Google Fonts: è l'unica richiesta esterna della pagina.
  Se preferisci non usarla, togli i tre `<link>` dei font in `index.html`: la pagina ripiega sui font di sistema.
- **Visualizzatore:** a tutto schermo, `←`/`→` cambiano versione, `Esc` chiude; `#nome-opera` nell'indirizzo apre
  direttamente un'opera. La sezione "Metodo" (`#metodo`) legge i dati: elenca le tecniche presenti nelle opere.
