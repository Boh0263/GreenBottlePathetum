import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from shapely.geometry import MultiPoint
import matplotlib.pyplot as plt


class ClusterEnvironment:
    def __init__(self):
        self.data = pd.DataFrame(columns=["x", "y", "label_cluster", "weight"])

    def update_environment(self, new_data, n_clusters=3, step_title="Algorithm Step"):
        """
        Aggiorna l'ambiente con nuovi dati, calcola i raggruppamenti KMeans,
        e visualizza i risultati.
        """
        if isinstance(new_data, list):
            new_data = pd.DataFrame(new_data)
        if not {"x", "y", "weight"}.issubset(new_data.columns):
            raise ValueError("Input data must include 'x', 'y' and 'weight' columns.")

        # Eseguiamo KMeans per il clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        new_data['label_cluster'] = kmeans.fit_predict(new_data[['x', 'y']])

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

        # Normalizzazione dei dati in base al peso
        weighted_data = data.copy()
        weighted_data[['x', 'y']] = weighted_data[['x', 'y']].multiply(weighted_data['weight'], axis=0)

        # Eseguiamo KMeans per il clustering sui dati ponderati
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(weighted_data[['x', 'y']])

        # Aggiungiamo le etichette di cluster ai dati
        data['label_cluster'] = labels

        # Visualizza il risultato del clustering in ogni passaggio
        self.cluster_env.update_environment(data, n_clusters=n_clusters,
                                            step_title=f"KMeans Clustering - {n_clusters} clusters")
        return labels, kmeans

    def calculate_silhouette(self, data, labels):
        """
        Calcoliamo il punteggio di silhouette per il clustering corrente.
        """
        score = silhouette_score(data[['x', 'y']], labels)
        return score

    def elbow_method(self, data, max_clusters=10):
        """
        Metodo della punta di gomito per trovare il miglior numero di cluster.
        """
        inertia_values = []
        for n_clusters in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(data[['x', 'y']])
            inertia_values.append(kmeans.inertia_)

        # Calcolare la seconda derivata dell'inertia
        second_derivative = np.diff(inertia_values, 2)

        # Trova il punto di gomito in cui la seconda derivata cambia significativamente
        elbow_point = np.argmax(np.abs(second_derivative)) + 2  # +2 perché diff riduce la lunghezza dell'array di 2

        # Plot dell'Elbow Method
        plt.plot(range(1, max_clusters + 1), inertia_values, marker='o', label="Inertia")
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.axvline(x=elbow_point, linestyle='--', color='r', label=f'Best number of clusters: {elbow_point}')
        plt.legend()
        plt.show()

        print(f"Numero ottimale di cluster trovato dal gomito: {elbow_point}")
        return elbow_point


# Esempio di utilizzo:

# Creiamo un set di dati casuali con pesi
np.random.seed(349)
n_points = 100
new_data = np.random.rand(n_points, 2) * 100  # 100 punti casuali nelle coordinate 0-100
weights = np.random.randint(1, 11, size=n_points)  # Assegniamo un peso tra 1 e 10 per ogni punto

# Creiamo il DataFrame
df_new_data = pd.DataFrame(new_data, columns=["x", "y"])
df_new_data["weight"] = weights

# Crea un'istanza di ClusterAnalysis
cluster_analysis = ClusterAnalysis(n_clusters=20)

# Metodo della punta di gomito per trovare il numero ottimale di cluster
best_n_clusters = cluster_analysis.elbow_method(df_new_data, max_clusters=10)

# Eseguiamo il clustering con il numero ottimale di cluster
labels, kmeans_model = cluster_analysis.perform_clustering(df_new_data, n_clusters=best_n_clusters)

# Calcoliamo il punteggio di silhouette per il clustering
silhouette_score = cluster_analysis.calculate_silhouette(df_new_data, labels)
print(f"Punteggio di silhouette con {best_n_clusters} cluster: {silhouette_score:.2f}")