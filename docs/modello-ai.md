# Modello AI e risultati

Il modello riconosce una cifra da una griglia di 8 × 8 valori. È una rete neurale con 64 ingressi, 24 neuroni nascosti ReLU e 10 uscite. Il JSON trasmesso occupa 3.266 byte e contiene architettura, pesi quantizzati a 8 bit, scale e bias.

Il ricevitore conosce il formato SIF1 e le operazioni necessarie per il calcolo. Riceve i pesi nei pacchetti, ricostruisce i 26 blocchi, verifica SHA-256 e carica il modello ricostruito. L'interprete gira sulla CPU con la libreria standard di Python e tratta il JSON esclusivamente come dati.

## Ripeti la prova

```text
python demo_ai.py
python demo_ai.py ricevi risultati/ai/ricevuto.bin
python demo_ai.py demo --solo-riparazione --output risultati/ai-riparazione
python demo_ai.py demo --perdita 1 --output risultati/ai-nessun-pacchetto
python demo_ai.py campagna
python -m unittest discover -s . -p "test_*.py"
```

La demo inclusa usa 20 esempi del test originale, due per cifra. Il comando `ricevi` usa il file con i pacchetti e le immagini di prova. Può classificare le immagini anche in una cartella priva del modello originale.

Per valutare tutte le 1.797 immagini di test, scarica `optdigits.tes` dalla pagina UCI e usa il percorso del file con `--test-set`.

```text
python demo_ai.py demo --test-set percorso/optdigits.tes
python demo_ai.py ricevi risultati/ai/ricevuto.bin --test-set percorso/optdigits.tes
```

## Risultati misurati

Nella prova a perdita casuale con probabilità impostata al 60% sono partiti 78 pacchetti. Ne sono andati persi 41 e altri 3 sono stati scartati dal CRC. I 34 pacchetti validi hanno fornito tutte le 26 equazioni indipendenti necessarie. Il modello ricostruito è identico byte per byte e ha dato le stesse predizioni dell'originale su tutte le 1.797 immagini. Le risposte corrette sono 1.727, pari al 96,10%.

Nella prova con perdita selettiva il ricevitore ha ricevuto solo la base di 26 pacchetti di riparazione. I 52 pacchetti scartati comprendevano tutti i pacchetti sorgente. Anche in questo caso il modello è stato ricostruito esattamente e ha prodotto le stesse 1.797 predizioni. La base scelta è invertibile per costruzione. Con 26 pacchetti arbitrari il recupero può fallire.

La campagna ha usato 24 semi per ciascun tasso di perdita, con il 5% di probabilità di corruzione fra i pacchetti arrivati.

| Probabilità di perdita | Recuperi esatti |
|---:|---:|
| 0% | 24/24 |
| 35% | 24/24 |
| 60% | 20/24 |
| 65% | 13/24 |
| 70% | 4/24 |
| 100% | 0/24 |

Questi conteggi descrivono il canale a file simulato. Quando il rango è insufficiente, il ricevitore richiede altri pacchetti e lascia l'inferenza spenta.

## Addestramento

Lo script `addestra_ai.py` richiede NumPy e i file UCI `optdigits.tra` e `optdigits.tes` nella stessa directory. Usa 3.823 immagini per addestrare la rete e il set originale di 1.797 immagini di altre persone per la sola valutazione. I pesi vengono quantizzati dopo l'addestramento. Le metriche e gli hash dei dati e del modello sono in `models/cifre-addestramento.json`.

```text
python addestra_ai.py percorso/dataset
```

## Dati e attribuzione

E. Alpaydin e C. Kaynak, *Optical Recognition of Handwritten Digits* (1998), UCI Machine Learning Repository, DOI `10.24432/C50P49`. Il file `models/cifre-esempi.json` ripubblica 20 righe di test, selezionate e riformattate in JSON. [Dataset UCI](https://doi.org/10.24432/C50P49) · [Licenza CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
