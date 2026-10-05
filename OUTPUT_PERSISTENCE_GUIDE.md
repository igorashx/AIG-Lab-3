# Ghid generic pentru salvarea output-ului unei rulări

Folosește acest document ca specificație pentru orice proiect care produce rezultate: antrenare ML, procesare de date, simulare, aplicație CLI sau pipeline ETL. Scopul este ca fiecare rulare să fie independentă, inspectabilă, reproductibilă și reluabilă.

## Principiul central

O rulare trebuie să aibă un director propriu, creat înainte de efectuarea lucrului. Nicio rulare nouă nu suprascrie artefactele unei rulări existente.

```text
<output-root>/
  run_<YYYYMMDD_HHMMSS_microseconds>/
    parameters.json
    metrics.json
    final_report.json
    execution.log
    checkpoints/
    samples/
    plots/
    exports/
```

Numele directorului trebuie să fie unic. Un timestamp cu microsecunde este suficient de obicei; pentru execuții concurente, se poate adăuga un identificator aleatoriu scurt. Toate căile artefactelor se construiesc într-un singur loc, într-o structură explicită precum `RunPaths`, nu ca șiruri de caractere împrăștiate prin cod.

## Contractul directoarelor

- `parameters.json`: contextul imuabil al rulării, salvat imediat după inițializare.
- `metrics.json`: istoricul actualizat incremental al progresului.
- `final_report.json`: rezumatul rezultatului final și al artefactului selectat.
- `execution.log`: jurnal textual, cu timp, nivel și mesaj.
- `checkpoints/`: stări complete de reluare.
- `samples/`: rezultate reprezentative intermediare sau finale, de exemplu imagini, fișiere CSV, documente sau payload-uri API.
- `plots/`: grafice și rapoarte vizuale.
- `exports/`: livrabile destinate consumului extern, dacă sunt necesare.

Creează numai directoarele de care proiectul are nevoie, însă păstrează aceleași responsabilități. Nu amesteca checkpoint-uri, loguri, grafice și fișiere de export în rădăcina proiectului sau într-un director comun tuturor rulărilor.

## Ciclul de viață al unei rulări

1. Validează configurația de intrare.
2. Creează directorul unic al rulării și subdirectoarele necesare.
3. Inițializează logger-ul către `execution.log`.
4. Salvează `parameters.json` înainte de lucrarea costisitoare.
5. Salvează artefacte și progres la intervale configurabile.
6. Actualizează atomic istoricul de metrici după fiecare unitate de progres.
7. Salvează checkpoint-ul curent și, separat, checkpoint-ul cel mai bun dacă există un criteriu de selecție.
8. La final, evaluează artefactul selectat și scrie raportul final.
9. Afișează și scrie în log calea directorului rulării.

Dacă rularea eșuează, directorul rămâne util: logul, configurația, metricile deja calculate și ultimul checkpoint trebuie să permită diagnosticarea sau reluarea.

## Manifestul inițial: `parameters.json`

Acest fișier explică exact ce s-a încercat. Include valori primitive JSON, nu obiecte dependente de runtime.

```json
{
  "started_at": "2026-10-05T14:30:00+00:00",
  "project": "numele-proiectului",
  "run_id": "run_20261005_143000_123456",
  "environment": {
    "python_version": "...",
    "library_versions": {"library": "..."},
    "device_or_platform": "..."
  },
  "reproducibility": {
    "seed": 42,
    "deterministic_mode": true,
    "input_snapshot": "opțional: hash, versiune sau cale"
  },
  "configuration": {
    "toate_opțiunile_efective": "valorile folosite în rulare"
  }
}
```

Reguli:

- Salvează configurația efectivă după aplicarea valorilor implicite și după validare.
- Convertește tipurile non-JSON în reprezentări explicite: căi în șiruri, enum-uri în valori, tuple în liste.
- Înregistrează versiuni de biblioteci, platforma și dispozitivul când acestea pot modifica rezultatul.
- Nu salva secrete, token-uri, parole sau date personale.
- Opțional, salvează versiunea Git (commit, branch și starea modificărilor) pentru cod care se află într-un repository.

## Metrici progresive: `metrics.json`

Păstrează istoricul sub formă de înregistrări, nu numai ultima valoare. O structură generală este:

```json
{
  "run_directory": "outputs/run_...",
  "metric_definitions": {"loss": "mai mic este mai bine"},
  "records": [
    {
      "step": 100,
      "elapsed_seconds": 12.5,
      "metrics": {"loss": 0.42, "accuracy": 0.81},
      "status": "completed"
    }
  ]
}
```

`step` poate fi epocă, batch, etapă de pipeline, iterație sau identificator de job. Folosește `null` pentru o metrică necalculată încă, nu o valoare inventată. Include definiția metricilor când numele lor pot fi ambigue.

Rescrie documentul complet prin înlocuire atomică: scrie întâi într-un fișier temporar din același director, apoi înlocuiește fișierul țintă. Astfel, întreruperea procesului nu lasă un JSON trunchiat.

```python
from pathlib import Path
import json

def save_json_atomic(path: Path, payload: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)
```

Pentru volume mari de date, scrie evenimente append-only în JSON Lines sau într-o bază de date, iar la final generează un rezumat JSON. Nu rescrie un JSON mare la fiecare pas.

## Checkpoint-uri și reluare

Un checkpoint pentru reluare trebuie să conțină tot ce este necesar pentru a continua corect, nu doar artefactul principal:

- starea artefactului: model, procesor, simulator sau pipeline;
- starea executorului: optimizator, cursor, coadă, parametri adaptați;
- progresul curent: epocă, pas sau etapă;
- istoricul necesar interfeței ori raportării;
- starea generatoarelor aleatoare, când reproducibilitatea contează;
- configurația efectivă sau o referință la manifest.

Folosește trei tipuri de checkpoint:

- `checkpoint_<pas>.ext`: instantanee periodice, păstrate pentru audit sau rollback;
- `latest.ext`: suprascris după fiecare salvare reușită, destinat reluării rapide;
- `best_<criteriu>.ext`: actualizat numai când se îmbunătățește criteriul de selecție.

Salvează checkpoint-uri după unități coerente de lucru, nu în mijlocul unei actualizări. La reluare, încarcă starea și verifică compatibilitatea configurației, a versiunilor și a datelor de intrare înainte de continuare.

## Artefacte vizuale și exporturi

Salvează artefacte reprezentative la același interval cu checkpoint-urile sau la o frecvență separată configurabilă. Numele trebuie să permită sortarea cronologică, de exemplu `epoch_003.png`, `step_000100.csv` sau `stage_02_report.html`.

Pentru comparații corecte, folosește intrări de evaluare fixe: același seed, aceleași eșantioane de test, același set de cereri sau aceeași configurație de randare. Salvează aceste intrări de control dacă nu pot fi regenerate sigur. Nu salva artefacte uriașe inutil; păstrează probe, agregări sau pointeri către stocarea externă.

## Raportul final

La terminare, selectează explicit rezultatul final: cel mai bun după o metrică de validare sau ultimul rezultat dacă nu există selecție. Nu presupune că ultima stare este cea mai bună.

```json
{
  "selected_artifact": "checkpoints/best_quality.ext",
  "selected_step": 42,
  "selection": {
    "criterion": "validation_loss",
    "direction": "minimize",
    "value": 0.18,
    "fallback_to_latest": false
  },
  "final_metrics": {"test_loss": 0.21},
  "finished_at": "2026-10-05T15:10:00+00:00",
  "status": "completed"
}
```

În caz de oprire controlată sau eroare, scrie un raport de stare cu `status` precum `stopped`, `failed` sau `completed`, fără a ascunde cauza în loguri.

## Logging

Folosește un logger per rulare, cu codare UTF-8 și format consecvent:

```text
2026-10-05 14:30:10 | INFO | Run started
2026-10-05 14:31:02 | INFO | Checkpoint saved: checkpoints/latest.ext
2026-10-05 15:10:00 | INFO | Run completed
```

În log se înregistrează începutul, configurația rezumată, progresul, salvările, avertismentele, erorile și calea directorului. Terminalul poate afișa aceleași mesaje, dar logul de pe disc rămâne sursa persistentă pentru depanare.

## Cerință gata de dat unui agent AI

```text
Implementează un sistem de persistare a output-ului bazat pe directoare izolate per rulare. Creează înainte de execuție `OUTPUT_ROOT/run_<timestamp_unic>/`, cu subdirectoarele potrivite pentru checkpoints, samples, plots și exports. Centralizează toate căile într-o structură RunPaths.

Salvează imediat un `parameters.json` UTF-8 cu configurația efectivă, timestamp-ul, mediul de execuție, versiunile relevante și informațiile de reproductibilitate. Menține un `metrics.json` cu istoricul complet al pașilor și actualizează-l atomic prin fișier temporar plus replace. Creează un `execution.log` per rulare.

La intervale configurabile, salvează artefacte reprezentative și checkpoint-uri complete pentru reluare. Menține `latest` pentru reluare rapidă, checkpoint-uri versionate pentru audit și `best_<criteriu>` când există un criteriu de calitate. La final, selectează explicit cel mai bun sau ultimul artefact, evaluează-l și scrie `final_report.json` cu selecția, metricile finale, statusul și timestamp-ul de finalizare. Nu suprascrie artefactele altei rulări și nu salva secrete în output.
```

## Listă de verificare

- Fiecare rulare are director propriu și unic.
- Toate căile sunt centralizate.
- Configurația efectivă și mediul sunt salvate înainte de procesare.
- Metricile păstrează istoric și sunt scrise atomic.
- Checkpoint-ul conține starea completă necesară reluării.
- Există separat `latest`, instantanee periodice și, când e cazul, cel mai bun rezultat.
- Logul, artefactele și raportul final sunt în același director de rulare.
- Raportul final indică explicit artefactul selectat și motivul selecției.
- Nu sunt persistate secrete sau date sensibile nejustificate.
