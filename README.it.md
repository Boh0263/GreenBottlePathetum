# GreenBottlePathetum

GreenBottlePathetum è un sistema basato sull'intelligenza artificiale sviluppato per ottimizzare i servizi di consegna nel contesto della logistica green. Questo progetto mira a supportare la piattaforma GreenBottle di AquaPure utilizzando algoritmi di clustering per assegnare efficacemente gli ordini ai corrieri, ottimizzare i percorsi e ridurre i costi operativi.

## Caratteristiche

- **Clustering degli Ordini**: Raggruppa gli ordini di consegna in cluster ottimali per l'assegnazione ai corrieri.
- **Ottimizzazione dei Percorsi**: Suggerisce i migliori percorsi di consegna per ciascun corriere.
- **Bilanciamento del Carico dei Corrieri**: Garantisce un carico di lavoro bilanciato tra i corrieri.
- **Scalabilità**: Gestisce efficientemente dataset di varie dimensioni e supporta diverse aree geografiche.

## Obiettivi del Progetto

GreenBottle utilizza veicoli elettrici per consegne sostenibili. Gli obiettivi del sistema includono:
- Distribuire gli ordini in modo ottimale tra i corrieri disponibili.
- Determinare i migliori percorsi per la consegna.
- Calcolare il numero ideale di corrieri necessario per evadere gli ordini.

L'approccio di clustering affronta il problema della categorizzazione degli ordini e del mapping dei percorsi ottimali, tenendo conto di efficienza e sostenibilità ambientale.

## Installazione

1. Clonare il repository:

   ```bash
   git clone https://github.com/Boh0263/GreenBottlePathetum.git
   ```

2. Accedere alla directory del progetto:

   ```bash
   cd GreenBottlePathetum
   ```

3. Installare le dipendenze richieste:

   ```bash
   pip install -r requirements.txt
   ```

## Test

Per testare il sistema, seguire questi passaggi:

1. Clonare il repository (se non già fatto):

   ```bash
   git clone https://github.com/Boh0263/GreenBottlePathetum.git
   ```

2. Accedere alla directory Testing:

   ```bash
   cd GreenBottlePathetum/Testing
   ```

3. Installare le dipendenze:

   ```bash
   pip install -r ../requirements.txt
   ```

4. Eseguire lo script di test:

   ```bash
   python main.py
   ```

Il sistema eseguirà un clustering di esempio su un dataset e mostrerà i risultati.

## Dettagli Tecnici

### Preprocessing dei Dati
I dati utilizzati includono:
- Informazioni sugli ordini (indirizzo, contenuto, ID ordine).
- Dati sui corrieri (numero e capacità).

Passaggi principali:
1. Mappatura degli indirizzi su un piano cartesiano per ottimizzare il calcolo dei percorsi.
2. Normalizzazione e riorganizzazione delle informazioni in formato strutturato.

### Algoritmi di Clustering
Gli algoritmi valutati includono:
- **K-Means**: Efficiente e ideale per un numero fisso di cluster.
- **Clustering Gerarchico**: Maggiore flessibilità, ma meno scalabile.
- **DBSCAN**: Adatto per rilevare outlier.

**Conclusione**: K-Means è il più adatto per questo progetto.

### Integrazione
Sono previsti miglioramenti futuri per integrare il sistema con la piattaforma AquaPure e fornire funzionalità aggiuntive come la selezione manuale degli ordini da clusterizzare.

## Miglioramenti Futuri
- Permettere agli admin di scegliere quali ordini processare nel sistema.
- Ottimizzare i percorsi in base a dati di traffico in tempo reale.

## Contributi
Contributi al progetto sono benvenuti. Effettuare un fork del repository e inviare una pull request con le modifiche.

## Licenza
Questo progetto è concesso sotto licenza MIT. Per ulteriori informazioni, consultare il file LICENSE.
