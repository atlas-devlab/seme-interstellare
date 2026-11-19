# Seme Interstellare

Seme Interstellare è un esperimento Python sulla trasmissione di programmi e modelli attraverso un canale con perdita di pacchetti. La prima demo trasporta un programma Fibonacci con la specifica della sua macchina. La seconda trasporta i pesi di un modello neurale che riconosce cifre scritte a mano.

## Demo AI

Su Windows apri `Avvia_demo_AI.cmd`. Da terminale, esegui `python demo_ai.py`. L'inferenza usa la sola libreria standard di Python. Per i risultati sull'intero test e le prove a diversi tassi di perdita, vedi [Modello AI e risultati](docs/modello-ai.md).

## Demo radio

La demo radio aggiunge modulazione FSK, rumore, interruzioni consecutive e un inviluppo a campana. Il ricevitore parte dai campioni in banda base e ricostruisce il modello dai pacchetti che superano il CRC. [Comandi, risultati e limiti](docs/radio.md)

## Avvio

Serve Python 3.10 o successivo. Su Windows puoi aprire `Avvia_demo.cmd`.

```text
python demo.py
python demo.py ricevi risultati/demo/ricevuto.bin
python verifica.py
```

Nella prova predefinita il mittente produce 66 pacchetti. Il canale ne perde 16 e ne altera 2. Il ricevitore ricostruisce 22 blocchi su 22 e verifica l'identità del messaggio prima di interpretare il programma. Il risultato parte da `0, 1, 1, 2, 3, 5, 8` e arriva a `610`.

Per cambiare i due valori iniziali o il canale, usa `python demo.py demo --help`. Per run paralleli usa cartelle di output distinte.

## Come funziona

1. `macchina.py` costruisce la capsula con bytecode, definizioni e vettori di prova.
2. `trasporto.py` divide la capsula in blocchi e aggiunge equazioni XOR di riparazione. Il ricevitore restituisce il messaggio quando le equazioni determinano tutti i blocchi e l'hash coincide.
3. `trame.py` ripete il manifest in ogni frame. Usa il CRC e il marcatore identico per riconoscere i frame validi anche dopo un'interruzione nel flusso.
4. `demo.py` genera il canale locale e controlla l'output con una seconda implementazione di Fibonacci.

Il ricevitore conosce già la codifica JSON, il framing e le operazioni della macchina. L'interprete limita istruzioni, stack e output. CRC e SHA-256 verificano l'integrità del contenuto ricevuto. L'identità del mittente richiederebbe un meccanismo di autenticazione separato.

Il canale è simulato tramite file. Per le misure storiche che hanno ispirato l'esperimento, vedi [Laboratorio Wow!](docs/laboratorio-wow.md).

## Licenza

GNU General Public License, versione 3, [`GPL-3.0-only`](LICENSE).
Gli esempi estratti dal dataset UCI sono attribuiti agli autori originali e distribuiti con licenza CC BY 4.0. [Attribuzione](docs/modello-ai.md#dati-e-attribuzione)
