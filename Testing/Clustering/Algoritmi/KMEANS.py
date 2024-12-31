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

        for n_clusters in range(2, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(data[['x', 'y']])
            inertia_values.append(kmeans.inertia_)

        # Plot Elbow Method
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, self.max_clusters + 1), inertia_values, marker='o', label="Inertia")
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.legend()

        # Indicate the optimal K with a red dotted line
        inertia_diff = np.diff(inertia_values)
        inertia_diff2 = np.diff(inertia_diff)
        elbow_index = np.argmax(inertia_diff2) + 2
        plt.axvline(x=elbow_index, color='red', linestyle='--', label=f"Optimal k: {elbow_index}")

        print(f"Optimal number of clusters (Elbow Method): {elbow_index}")
        plt.show()

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
