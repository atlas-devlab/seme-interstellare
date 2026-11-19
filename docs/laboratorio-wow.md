# Laboratorio Wow!

L'indagine parte da `6EQUJ5`, le sei classi di intensità pubblicate per il segnale Wow! registrato da Big Ear. Il rapporto di Jerry Ehman spiega che la registrazione conservava un dato per canale ogni 12 secondi. Questa granularità lascia indeterminata la modulazione più rapida. [Rapporto di Big Ear](https://www.bigear.org/Wow30th/wow30th.htm)

Abbiamo provato 42 ricodifiche definite da noi, per un totale di 2.688 viste in byte. Abbiamo poi applicato ripetizione e Hamming a quelle ricodifiche, ottenendo 1.680 candidati e 26.880 viste in byte. La ricerca di ZIP, gzip e zlib ha restituito zero contenuti con integrità verificata. Le corrispondenze sintattiche di DEFLATE grezzo erano prive di checksum. In una prova separata su simboli GF(32), nessuna delle 200 configurazioni concordava con tutte e sei le misure.

Un controllo con pacchetti XOR costruiti in laboratorio ha invece ricostruito integralmente un archivio noto. Questa prova ha dato origine a Seme Interstellare, dove il messaggio porta anche un programma. La catalogazione delle ricodifiche misura quanto regge quel repertorio di ipotesi sulle sei classi disponibili. Le misure conservate non hanno permesso di ricostruire un programma dal segnale Wow!.
