# Seme Interstellare via radio simulata

La demo modula i pacchetti della rete neurale con due toni FSK a ±3.000 Hz in banda base. Usa 6.000 simboli al secondo e campiona I e Q a 48.000 campioni al secondo. La finestra dura 72 secondi. Un inviluppo gaussiano rappresenta la variazione di sensibilità del fascio durante il passaggio. I campioni attraversano un filtro passa basso con tagli nominali a ±5 kHz. La specifica riguarda una simulazione locale senza trasmissione RF.

Il ricevitore conosce la modulazione nominale e il preambolo. Scorre gli otto allineamenti possibili per ogni simbolo e cerca i preamboli. I frame con CRC valido passano al decoder XOR, che usa 26 equazioni indipendenti per ricostruire i blocchi e verifica SHA-256 prima di avviare l’inferenza. Il ricevitore parte dai campioni I/Q e non riceve i parametri casuali del trasmettitore, i tempi dei buchi, gli offset dei pacchetti o i pesi originali.

## Ripeti la prova

Installa le dipendenze in un ambiente Python e usa i comandi dalla cartella del progetto.

```text
python -m pip install -r requirements-radio.txt
python radio.py prova --salva-iq --grafico
python radio.py ricevi risultati/radio/ricevuto_iq.npy
python radio.py prova --snr 0 --output risultati/radio-forte-rumore
python radio.py controllo-medie
python -m unittest discover -s . -p "test_*.py"
```

La prova standard usa i 20 esempi UCI inclusi. Aggiungi `--test-set percorso/optdigits.tes` a `prova` o `ricevi` per valutare il set UCI completo. [Modello e attribuzione dei dati](modello-ai.md)

## Risultati

Tutte le prove usano 78 pacchetti e uno scarto fisso di frequenza di 140 Hz. I quattro vuoti, per un totale di 16,7 secondi, rimuovono il segnale utile in tratti consecutivi. Nelle prove rumorose il ricevitore vede solo rumore in quei tratti. L’SNR è il rapporto tra il picco di potenza del segnale e la potenza del rumore complesso per campione prima del filtro. Non usa la scala stampata dal Big Ear.

| Canale simulato | Frame con CRC valido | Esito |
|---|---:|---|
| Senza rumore né vuoti | 78/78 | Recupero esatto |
| Quattro vuoti, senza rumore | 57/78 | Recupero esatto |
| Quattro vuoti, rumore con SNR di picco 8 dB | 40/78 | Recupero esatto |
| Quattro vuoti, rumore con SNR di picco 0 dB | 0/78 | Inferenza non avviata |

Nella prova a 8 dB il modello da 3.266 byte è identico all’originale e ha prodotto le stesse 1.797 predizioni sul test UCI, con 1.727 cifre corrette (96,10%). Questi risultati misurano gli scenari scelti. L’esperimento usa un orologio di simbolo fisso e non ricostruisce derive di frequenza, fading realistico o interferenze terrestri.

## Formato binario compatto

Il nuovo formato SIC1 invia i blocchi come byte e codifica le intestazioni con campi binari. L’ID identifica i blocchi sorgente, quindi per questi non serve inviare la maschera. Ogni pacchetto conserva un CRC e ogni trasmissione conserva l’hash SHA-256 del modello. L’informazione contenuta nelle 78 equazioni di riparazione e sorgente resta identica.

| Misura sulla stessa finestra di 72 s | SIF1 (JSON) | SIC1 (binario) |
|---|---:|---:|
| Byte inviati, preamboli e lunghezze radio inclusi | 41.545 | 15.262 |
| Tempo di modulazione, escluse le pause | 55,39 s | 20,35 s |
| Pacchetti validi | 40 | 42 |
| Modello ricostruito | Identico | Identico |

Le due prove trasmettono gli stessi 78 ID agli stessi istanti, con gli stessi campioni di rumore, lo stesso scarto di frequenza e i medesimi quattro buchi temporali. La baseline SIF1 è risultata identica campione per campione alla precedente registrazione simulata. La durata più breve dei frame cambia quali pacchetti superano il CRC. Per controllare che il risultato non dipenda dai nuovi ID salvati, abbiamo tenuto solo i 38 ID validi in entrambe le prove. Anche questo sottoinsieme ha ricostruito il modello byte per byte e ha prodotto tutte le stesse 1.797 predizioni.

```text
python confronta_radio.py --salva-iq
python radio.py ricevi risultati/radio-compatto/ricevuto_compatto_iq.npy
python radio.py prova --compatto --output risultati/radio-binario
```

Il confronto usa di default i 20 esempi forniti. `--test-set percorso/optdigits.tes` consente di usare il test completo. `--riferimento-iq percorso/ricevuto_iq.npy` verifica la baseline con una registrazione SIF1 precedente, se disponibile.

## Collegamento con il Wow!

La simulazione conserva sia i campioni I/Q sia sei medie di potenza da 10 secondi distanziate di 12 secondi, secondo la cadenza descritta nel rapporto storico. Le medie del programma usano unità relative e una sottrazione del rumore nota. La curva a campana è un’ipotesi del canale. [Rapporto Big Ear](https://www.bigear.org/Wow30th/wow30th.htm)

In un controllo separato due sequenze BPSK sintetiche differiscono in 712 simboli su 1.440 ma hanno le stesse sei medie di potenza. La ricostruzione della rete si basa sui campioni I/Q simulati. Le sei classi pubblicate per il Wow! non conservano quei campioni.
