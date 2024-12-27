import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from shapely.geometry import MultiPoint
import matplotlib.pyplot as plt

class ClusterEnvironment:
    def __init__(self):
        self.data = pd.DataFrame(columns=["x", "y", "label_cluster"])

    def update_environment(self, new_data, n_clusters=3, step_title="Algorithm Step"):
        """
        Aggiorna l'ambiente con nuovi dati, calcola i raggruppamenti KMeans,
        e visualizza i risultati.
        """
        if isinstance(new_data, list):
            new_data = pd.DataFrame(new_data)
        if not {"x", "y"}.issubset(new_data.columns):
            raise ValueError("Input data must include 'x' and 'y' columns.")

        # Eseguiamo KMeans per il clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        new_data['label_cluster'] = kmeans.fit_predict(new_data[['x', 'y']])

        # Calcoliamo il punteggio di silhouette
        silhouette_avg = silhouette_score(new_data[['x', 'y']], new_data['label_cluster'])
        print(f"{step_title}: Silhouette Score per {n_clusters} cluster: {silhouette_avg:.2f}")

        # Aggiorniamo i dati dell'ambiente
        self.data = new_data.copy()
        self._visualize(step_title)

    def _visualize(self, title="Clustered Data"):
        """
        Visualizza i punti e i cluster usando Plotly, includendo i poligoni
        dei cluster e i punti colorati.
        """
        cluster_polygons = []
        for cluster_label in self.data["label_cluster"].unique():

            # Filtra i punti del cluster corrente
            cluster_points = self.data[self.data["label_cluster"] == cluster_label][["x", "y"]]

            # Crea il concavità dei punti (convex hull) del cluster
            multipoint = MultiPoint(cluster_points.values)
            convex_hull = multipoint.convex_hull

            # Gestiamo se il convex hull è un Polygon o LineString
            if convex_hull.geom_type == 'Polygon':
                hull_coords = list(convex_hull.exterior.coords)
            elif convex_hull.geom_type == 'LineString':
                hull_coords = list(convex_hull.coords)
            else:
                print(f"Unexpected geometry type: {convex_hull.geom_type}")
                continue

            # Dividiamo le coordinate del convex hull
            hull_x = [coord[0] for coord in hull_coords]
            hull_y = [coord[1] for coord in hull_coords]

            # Aggiungiamo il poligono del cluster
            cluster_polygons.append(go.Scatter(
                x=hull_x,
                y=hull_y,
                fill="toself",  # Riempiamo l'area del cluster
                fillcolor=f"rgba({np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {np.random.randint(0, 255)}, 0.3)",
                line=dict(width=2, color='rgba(0,0,0,0)'),  # Senza bordo
                name=f"Cluster {cluster_label}"
            ))

        # Crea un grafico a dispersione per i punti
        scatter = go.Scatter(
            x=self.data["x"],
            y=self.data["y"],
            mode="markers",
            marker=dict(
                size=10,
                color=self.data["label_cluster"],  # Colore in base al cluster
                colorscale="Jet",  # Scala di colori
                opacity=0.6  # Trasparenza per i punti
            ),
            text=self.data["label_cluster"],  # Mostriamo l'etichetta del cluster al passaggio del mouse
            hoverinfo="text+x+y"  # Mostriamo x, y e l'etichetta del cluster
        )

        # Impostiamo il layout del grafico
        layout = go.Layout(
            title=title,
            xaxis=dict(title="X", range=[0, 100]),  # Intervallo X
            yaxis=dict(title="Y", range=[0, 100]),  # Intervallo Y
            showlegend=True
        )

        # Creiamo il grafico e lo mostriamo
        fig = go.Figure(data=[scatter] + cluster_polygons, layout=layout)
        fig.show()


class ClusterAnalysis:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.cluster_env = ClusterEnvironment()  # Creiamo l'ambiente per la visualizzazione dei cluster

    def perform_clustering(self, data, n_clusters=None):
        """
        Eseguiamo il clustering con KMeans e aggiorniamo l'ambiente per ogni passaggio.
        """
        if n_clusters is None:
            n_clusters = self.n_clusters  # Usa il numero di cluster di default

        # Eseguiamo KMeans per il clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data[['x', 'y']])

        # Aggiungiamo le etichette di cluster ai dati
        data['label_cluster'] = labels

        # Calcoliamo il punteggio di silhouette per il clustering
        silhouette_avg = silhouette_score(data[['x', 'y']], labels)
        print(f"Silhouette Score per {n_clusters} cluster: {silhouette_avg:.2f}")

        # Visualizza il risultato del clustering
        self.cluster_env.update_environment(data, n_clusters=n_clusters,
                                            step_title=f"KMeans Clustering - {n_clusters} clusters")
        return labels, kmeans, silhouette_avg

    def elbow_method(self, data, max_clusters=10):
        """
        Metodo della punta di gomito per trovare il miglior numero di cluster.
        """
        inertia_values = []  # Lista per salvare i valori di inerzia

        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(data[['x', 'y']])
            inertia_values.append(kmeans.inertia_)

        # Troviamo l'elbow (il punto di inflection) dove la riduzione dell'inerzia inizia a rallentare
        inertia_diff = np.diff(inertia_values)  # Calcoliamo la differenza tra i valori di inerzia successivi
        inertia_diff2 = np.diff(inertia_diff)  # Calcoliamo la seconda derivata per trovare il punto di rallentamento

        # L'indice del massimo valore della seconda derivata è il nostro elbow
        elbow_index = np.argmax(inertia_diff2) + 2  # Offset di 2 per l'indice

        print(f"Numero ottimale di cluster basato sul metodo del gomito: {elbow_index}")

        # Plot dei valori di inerzia per il metodo del gomito
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, max_clusters + 1), inertia_values, marker='o', label="Inertia")
        plt.axvline(elbow_index, color='red', linestyle='--', label=f"Numero Ottimale (Elbow): {elbow_index}")
        plt.title('Metodo della Punta di Gomito')
        plt.xlabel('Numero di Cluster (k)')
        plt.ylabel('Inertia')
        plt.legend()
        plt.show()

        print("Metodo della punta di gomito completato.")
        return elbow_index


# Esempio di utilizzo:

# Creiamo un set di dati casuali
np.random.seed(240)
n_points = 100
new_data = np.random.rand(n_points, 2) * 100  # 100 punti casuali nelle coordinate 0-100
df_new_data = pd.DataFrame(new_data, columns=["x", "y"])

# Crea un'istanza di ClusterAnalysis
cluster_analysis = ClusterAnalysis(n_clusters=25)

# Metodo della punta di gomito per trovare il numero ottimale di cluster
best_n_clusters = cluster_analysis.elbow_method(df_new_data, max_clusters=25)

# Eseguiamo il clustering con il numero ottimale di cluster
labels, kmeans_model, silhouette_avg = cluster_analysis.perform_clustering(df_new_data, n_clusters=best_n_clusters)

# Calcoliamo e stampiamo il punteggio di silhouette finale
print(f"Punteggio di silhouette finale con {best_n_clusters} cluster: {silhouette_avg:.2}")


