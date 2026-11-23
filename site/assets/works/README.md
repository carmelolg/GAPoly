# Opere

Ogni opera è **una foto** con una o più **versioni** (una per forma o per esecuzione). Il sito si rigenera da solo a
ogni pubblicazione: basta aggiungere o togliere cartelle.

## Come si chiama la cartella dell'opera

`00_nome-opera`, per esempio `01_artemisia`: il numero è la posizione in galleria (vedi "Ordine della galleria")
e `nome-opera` diventa il titolo se non ne scrivi uno. La data si legge da `date` in `meta.json`.
Le cartelle che iniziano con `_` o `.` vengono ignorate (come `_template`).

## Struttura

```
01_tokyo/
  original.jpg      la foto di partenza, condivisa da tutte le versioni
  meta.json         titolo, descrizione, tecnica, data, tag, "main" (la versione di copertina)
  disc/             una cartella per versione: copia qui la cartella --out di gapoly
    final.png
    evolution.gif
    history.csv
    meta.json       forma, etichetta, parametri di questa versione (opzionale)
  mosaic/
    final.png  evolution.gif  history.csv
  polygon-hq/       il nome è libero: due esecuzioni con la stessa forma possono convivere
    final.png  evolution.gif  history.csv
```

- Una **versione** è una sottocartella con almeno `final.png` (o `evolution.gif`). Le altre sottocartelle vengono ignorate.
- Il sito mostra le versioni come pulsanti nel dettaglio dell'opera, più la vista **Tutte le versioni** (tutte
  affiancate alla foto originale).
- **Opera con una versione sola:** metti `final.png`, `evolution.gif` e `history.csv` direttamente nella cartella
  dell'opera, senza sottocartelle. Si possono anche avere i due modi insieme.
- **Copertina (anteprima):** in automatico è la versione con la fitness migliore (media di `best_mse` e `mean_mse`
  all'ultima generazione, da `history.csv`). Per forzarne un'altra: `"main": "mosaic"` in `meta.json` (nome della sottocartella).
- **Opera in evidenza nella home:** `"featured": true` nel `meta.json`.
- **Ordine della galleria:** `"order": n` nel `meta.json` (più piccolo = prima). Le opere con `"pinned": true` vengono
  prima di tutte, in ordine di `order`; le altre seguono in ordine di `order`, e quelle senza `order` restano in
  fondo, dalla più recente. Per nascondere un'opera, rinomina la cartella con `_` iniziale.

## Come si scrive la scheda di un'opera (`meta.json`)
Segue la convenzione delle didascalie dei musei: il titolo da solo, l'autore e l'anno a parte.

| campo | esempio | note |
|---|---|---|
| `title` | `"Donna con coppa"` | solo il titolo, senza autore né anno; tieni il titolo originale |
| `artist` | `"Vincent van Gogh"` | chi ha fatto l'opera mostrata (il dipinto, la foto); se non è certo: `"Attribuito a Guido Reni"` (o `{"it": ..., "en": ...}`) |
| `year` | `"1638–1639"` | quando è stata fatta l'opera originale, non la data dell'elaborazione |
| `also_known_as` | `["Artemisia"]` | altri titoli con cui è nota: compaiono nel visualizzatore |
| `location` | `"Giardini Reali, Torino"` | dove si trova l'opera originale (per la foto di un monumento) |
| `credits` | `[{"label": "Foto su Unsplash", "url": "https://..."}]` | righe di crediti; il link è facoltativo e solo `http(s)` |
| `date` | `"2026-11-19"` | la data dell'**elaborazione** (si ricava dal nome della cartella) |
| `description`, `tags`, `technique` | | facoltativi, anche per lingua |

La scheda mostra "autore, anno" sotto il titolo; l'elaborazione è indicata a parte ("Elaborazione: 19 novembre 2026").
Se non conosci un dato lascialo fuori: il sito non scrive nulla al suo posto.
Per la foto di un'opera, `artist` è l'autore dell'opera (lo scultore, il pittore) e il fotografo va in `credits`;
per una foto presa da un archivio (per esempio Unsplash) `artist` è il fotografo. "Attribuito a ..." solo quando
l'attribuzione è davvero incerta.

## File di una versione

| file | obbligatorio | cosa è |
|---|---|---|
| `final.png` | sì (o una `evolution.gif`) | il risultato |
| `evolution.gif` (o `evolution.webp`) | no | l'evoluzione: compare con il pulsante "Guarda l'evoluzione" |
| `history.csv` | no | l'errore per generazione: dà le statistiche e il grafico |
| `meta.json` | no | `shape`, `label`, `params` (vedi `_template/disc/meta.json`) |

La **forma** si riconosce dal nome della sottocartella se inizia con una forma di gapoly (`disc`, `polygon-hq`,
`mosaic`...); altrimenti scrivila in `meta.json`. L'**etichetta** del pulsante è il nome della forma
(`disc` diventa "Dischi") o il nome della cartella.

La cartella `snapshots/` di gapoly non serve, non copiarla.
La foto di partenza può chiamarsi `original.jpg` (anche `.png`, `.webp`).

## Peso dei file

Il sito pubblica versioni leggere (immagini ridotte, GIF a 640 px). I file originali restano nel repository:
GitHub avvisa sopra i 50 MB per file e rifiuta sopra i 100 MB. Una `evolution.gif` di gapoly può pesare 25-40 MB:
con più versioni per opera il peso cresce in fretta; se vuoi tenere il repository leggero, riducile prima di copiarle.

## Altre tecniche (automi cellulari, ecc.)

Stessa cartella, stessa struttura. Servono solo `final.png` (o `evolution.gif`) e un `meta.json` con
`"technique": "Automa cellulare"`: la tecnica diventa un filtro nella galleria.
