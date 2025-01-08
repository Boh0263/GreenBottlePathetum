import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from Testing.Dataset import random_points_generator as rpg
from Testing.Clustering.PathetumEnviroment import ClusterEnvironment
from Testing.Clustering.ClusterAnalysis import ClusterAnalysis


class KMeansAnalysis(ClusterAnalysis):

    def __init__(self, max_clusters=10, n_clusters=3, cluster_env=None):
        super().__init__()
        self.max_clusters = max_clusters
        self.n_clusters = n_clusters
        self.cluster_env = cluster_env if cluster_env else ClusterEnvironment()

    def run_kmeans(self, data, n_clusters):
        """
        Run KMeans with a given number of clusters and return the model and the data with cluster labels.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=rpg.get_random_seed())
        data['label_cluster'] = kmeans.fit_predict(data[['x', 'y']])
        return kmeans, data

    def perform_clustering(self, data, use_elbow=True):
        """
        Perform KMeans clustering using the optimal number of clusters determined by
        the Elbow Method or Silhouette Method.
        """
        # Determine the optimal number of clusters
        optimal_k = self.find_optimal_k(data, use_elbow)

        # Perform KMeans with the optimal k
        kmeans, clustered_data = self.run_kmeans(data, optimal_k)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(data[['x', 'y']], clustered_data['label_cluster'])
        print(f"Silhouette Score for {optimal_k} clusters: {silhouette_avg:.2f}")

        # Visualize results using ClusterEnvironment
        self.cluster_env.update_environment(clustered_data, n_clusters=optimal_k,
                                            step_title=f"KMeans Clustering - {optimal_k} clusters")

        return kmeans, silhouette_avg

    def find_optimal_k(self, data, use_elbow=True):

        if use_elbow:
            optimal_k = self.elbow_method(data)
        else:
            optimal_k = self.silhouette_method(data)

        return optimal_k

    def elbow_method(self, data):
        inertia_values = []

        # Calcolo dell'inerzia per diversi valori di k
        for n_clusters in range(2, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(data[['x', 'y']])
            inertia_values.append(kmeans.inertia_)

        # Calcolo della differenza assoluta tra valori consecutivi di inerzia
        inertia_diff = np.abs(np.diff(inertia_values))

        # Trovare il punto di piega come il primo valore di k dove la riduzione rallenta
        elbow_index = np.argmax(inertia_diff < 0.1 * inertia_diff[0]) + 2

        # Tracciamento del grafico
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, self.max_clusters + 1), inertia_values, marker='o', label="Inertia")
        plt.axvline(x=elbow_index, color='red', linestyle='--', label=f"Optimal k: {elbow_index}")
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.legend()
        plt.show()

        print(f"Optimal number of clusters (Elbow Method): {elbow_index}")
        return elbow_index

    def refined_elbow_method(self, data, tolerance=1000):
        inertia_values = []

        # Initial computation of inertia for range of cluster counts
        for n_clusters in range(2, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(data[['x', 'y']])
            inertia_values.append(kmeans.inertia_)

        def find_elbow(inertia, tolerance):
            """Finds elbow points by analyzing drops in inertia."""
            inertia_diff = np.abs(np.diff(inertia))
            elbow_indices = []

            for i in range(1, len(inertia_diff)):
                if inertia_diff[i] < tolerance * inertia_diff[0]:
                    elbow_indices.append(i + 1)

            return elbow_indices

        elbow_indices = find_elbow(inertia_values, tolerance)

        # Iterative refinement of elbow selection
        refined_elbow = None
        while len(elbow_indices) > 1:
            sub_range = list(range(elbow_indices[0] + 2, elbow_indices[-1] + 3))  # Adjust range to avoid overlap
            inertia_values_sub = []

            for n_clusters in sub_range:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                kmeans.fit(data[['x', 'y']])
                inertia_values_sub.append(kmeans.inertia_)

            elbow_indices = find_elbow(inertia_values_sub, tolerance)

            if len(elbow_indices) == 1:
                refined_elbow = sub_range[elbow_indices[0] - 1]

        # Final optimal number of clusters
        elbow_index = elbow_indices[0] if elbow_indices else (refined_elbow if refined_elbow else 2)

        # Plot the inertia values and the detected elbow point
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, min(self.max_clusters + 1, len(data))), inertia_values, marker='o', label="Inertia")
        plt.axvline(x=elbow_index, color='red', linestyle='--', label=f"Optimal k: {elbow_index}")
        plt.title('Refined Elbow Method')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.legend()
        plt.show()

        print(f"Refined optimal number of clusters: {elbow_index}")
        return elbow_index

    def silhouette_method(self, data):
        silhouette_scores = []

        for n_clusters in range(2, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(data[['x', 'y']])
            silhouette_avg = silhouette_score(data[['x', 'y']], labels)
            silhouette_scores.append(silhouette_avg)

        # Plotting silhouette scores
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, self.max_clusters + 1), silhouette_scores, marker='o', label="Silhouette Score")
        plt.title('Silhouette Method')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.legend()

        # Determine the optimal k and silhouette score
        optimal_k = np.argmax(silhouette_scores) + 2  # +2 because range starts from 2
        optimal_silhouette = silhouette_scores[optimal_k - 2]  # Index adjustment

        # Draw vertical line at the optimal k
        plt.axvline(x=optimal_k, color='red', linestyle='--', label=f"Optimal k: {optimal_k}")

        # Draw horizontal line at the silhouette score for the optimal k
        plt.axhline(y=optimal_silhouette, color='red', linestyle='--',
                    label=f"Silhouette Score: {optimal_silhouette:.2f}")

        print(f"Optimal number of clusters (Silhouette Method): {optimal_k}")
        plt.show()

        return optimal_k
