# Laborator 3 — MLP-GAN pentru generarea cifrelor MNIST

## Obiectiv

Acest laborator construiește un **Generative Adversarial Network (GAN)** în PyTorch pentru a genera imagini noi, de $28 \times 28$ pixeli, asemănătoare cifrelor manuscrise din setul MNIST.

Modelul este compus din două rețele care se antrenează competitiv:

- **Generatorul** primește zgomot aleator $z$ și produce o imagine sintetică.
- **Discriminatorul** primește o imagine și estimează dacă aceasta este reală sau generată.

Generatorul încearcă să păcălească Discriminatorul, iar Discriminatorul încearcă să distingă cât mai bine imaginile reale de cele sintetice. În acest laborator se folosesc rețele complet conectate (MLP), pentru o implementare concisă și ușor de urmărit.

Generatorul folosește `Linear → BatchNorm1d → LeakyReLU(0.2)` după fiecare strat liniar ascuns și `Tanh` numai la ieșire. Discriminatorul și clasificatorul auxiliar pentru metrici folosesc `LeakyReLU(0.2)`, evitând activările ReLU simple care pot produce neuroni inactivi.

## Structură

| Element | Rol |
| --- | --- |
| `gan_mnist.py` | Script de antrenare, generare, vizualizare și reluare. |
| `data/` | Datele MNIST descărcate automat. |
| `outputs/run_YYYYMMDD_HHMMSS_ffffff/` | Director izolat pentru o singură execuție. |
| `.../samples/` | Grile PNG cu cifre generate la diferite epoci. |
| `.../plots/` | Dashboard-uri Plotly interactive pentru pierderi și metrici de calitate. |
| `.../checkpoints/latest.pt` | Ultima stare completă a antrenării din acea execuție. |
| `.../metrics.json` | Metrici de antrenare, FID și IS, actualizate după fiecare epocă. |
| `.../parameters.json` | Configurația, device-ul și versiunile folosite la pornire. |
| `.../execution.log` | Jurnalul persistent al execuției. |
| `.../fixed_noise.pt` | Zgomotul latent fix folosit pentru toate grilele de mostre. |
| `.../best_model_metrics.json` | Toate metricile train/validation/test pentru modelul ales de early stopping. |

## Instalare

În PowerShell, activați mediul virtual și instalați dependențele:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Pentru PyTorch cu suport CUDA, instalați înainte versiunea potrivită pentru placa video prin selectorul oficial PyTorch. Scriptul selectează automat CUDA dacă este disponibilă; în caz contrar folosește CPU.

## Rulare

Testați componentele fără descărcarea MNIST:

```powershell
python gan_mnist.py --smoke-test
```

Porniți antrenarea standard:

```powershell
python gan_mnist.py --epochs 30 --batch-size 128
```

Configurați independent adâncimea și numărul de neuroni ai fiecărui strat ascuns prin liste separate de valori. De exemplu, Generatorul va avea trei straturi ascunse de 128, 256 și 512 neuroni, iar Discriminatorul două straturi de 512 și 128 neuroni:

```powershell
python gan_mnist.py --generator-layers 128,256,512 --discriminator-layers 512,128
```

Fiecare valoare separată prin virgulă reprezintă un strat ascuns. Valorile implicite sunt `256,512` pentru Generator și `512,256` pentru Discriminator.

Pentru un test rapid de flux, limitați batch-urile:

```powershell
python gan_mnist.py --epochs 1 --max-batches 10 --save-interval 1 --device cpu
```

Reluați din ultimul checkpoint, indicând numărul total de epoci dorit:

```powershell
python gan_mnist.py --epochs 50 --resume outputs\run_YYYYMMDD_HHMMSS_ffffff\checkpoints\latest.pt
```

Folosiți `python gan_mnist.py --help` pentru toate opțiunile, inclusiv dimensiunea vectorului latent, rata de învățare, numărul workerilor și directoarele de ieșire.

## Monitorizare în terminal

Antrenarea utilizează bare de progres `tqdm` pentru epoci și mini-batch-uri. Bara internă actualizează:

- `D_loss` — pierderea Discriminatorului;
- `G_loss` — pierderea Generatorului;
- `D_real` — probabilitatea medie atribuită imaginilor reale;
- `D_fake` — probabilitatea medie atribuită imaginilor false.

La finalul fiecărei epoci este afișat un rezumat scurt, iar salvările sunt raportate fără a fragmenta barele de progres.

### Metrici de calitate

La fiecare epocă, scriptul evaluează imaginile generate față de ambele partiții MNIST:

- `FID train` și `FID validation` — distanța Fréchet față de imaginile reale; **mai mic este mai bine**;
- `IS train` și `IS validation` — Inception Score; **mai mare este mai bine**.
- `precision` — proporția imaginilor generate apropiate de manifold-ul real; **mai mare este mai bine**;
- `recall` — acoperirea manifold-ului real de către imaginile generate; **mai mare este mai bine**.

Pentru MNIST, aceste valori sunt calculate cu un clasificator CNN antrenat doar pe cifre MNIST, nu cu modelul ImageNet Inception-v3. Astfel, metricile sunt relevante pentru imagini de $28 \times 28$ și pentru cele zece cifre. Evaluatorul este antrenat o dată la începutul rulării și rămâne înghețat în timpul antrenării GAN.

Implicit sunt evaluate 512 imagini din fiecare partiție. Pentru o rulare mai rapidă se poate reduce numărul, iar frecvența se poate modifica astfel:

```powershell
python gan_mnist.py --metric-samples 256 --metrics-every 5
```

Chiar dacă FID/IS sunt evaluate mai rar, `metrics.json` este actualizat după fiecare epocă; epocile fără evaluare au câmpul `quality_metrics` setat la `null`.

Precision și recall sunt salvate întotdeauna în `metrics.json`, însă apar în sumarul terminal numai la cerere:

```powershell
python gan_mnist.py --show-precision-recall
```

## Mostre cu zgomot fix

Fiecare rulare creează un singur tensor latent, salvat în `fixed_noise.pt`. Același tensor este utilizat pentru toate fișierele `samples/epoch_XXX.png`, deci fiecare poziție din grilă reprezintă aceeași intrare latentă la toate epocile. Astfel poate fi urmărită vizual evoluția Generatorului.

## Dashboard-uri interactive

Fișierele `plots/losses.html` și `plots/quality_metrics.html` sunt dashboard-uri Plotly autonome. Deschideți-le în browser pentru zoom, pan, selectarea seriilor și inspectarea valorilor la hover.

## Early stopping

Early stopping este dezactivat implicit. El monitorizează metricile calculate pe **validation** și funcționează la fiecare evaluare de calitate, nu neapărat la fiecare epocă dacă `--metrics-every` este mai mare decât 1.

Exemple:

```powershell
# Oprește dacă validation FID nu se micșorează timp de 8 evaluări.
python gan_mnist.py --early-stopping-metric fid --early-stopping-patience 8

# Oprește dacă validation IS nu crește cu cel puțin 0.02 timp de 10 evaluări.
python gan_mnist.py --early-stopping-metric is --early-stopping-patience 10 --early-stopping-min-delta 0.02
```

Cel mai bun model după metrica aleasă este salvat ca `checkpoints/best_quality.pt`.

La final, acest checkpoint este reîncărcat și evaluat pe trei partiții distincte: train, validation și test. Raportul complet FID, IS, precision și recall este scris în `best_model_metrics.json`. Validation este o partiție deterministă de 10.000 exemple extrasă din MNIST train cu seed-ul rulării; test este setul MNIST oficial, care nu este folosit pentru antrenarea GAN sau pentru alegerea early stopping.

## Interpretarea rezultatelor

Pierderile GAN nu trebuie interpretate ca în clasificarea clasică: ele pot oscila chiar și când imaginile devin mai bune. Criteriul principal este evoluția grilelor generate cu același zgomot fix, salvate în `outputs/samples/`. După mai multe epoci, structurile ar trebui să semene progresiv cu cifrele MNIST.

Un MLP GAN este limitat: nu exploatează explicit structura spațială a imaginii și poate produce cifre mai puțin clare decât un DCGAN convoluțional. Totuși, el evidențiază direct mecanismul adversarial și permite testarea rapidă a experimentelor.

## Artefacte produse

- Mostre: `outputs/run_.../samples/epoch_XXX.png`;
- Grafice Plotly: `outputs/run_.../plots/losses.html` și `quality_metrics.html`;
- Metrici actualizate după fiecare epocă: `outputs/run_.../metrics.json`;
- Parametrii rulării: `outputs/run_.../parameters.json`;
- Log persistent: `outputs/run_.../execution.log`;
- Zgomot latent fix: `outputs/run_.../fixed_noise.pt`;
- Raportul final al modelului optim: `outputs/run_.../best_model_metrics.json`;
- Checkpoint-uri: `outputs/run_.../checkpoints/checkpoint_epoch_XXX.pt` și `latest.pt`.