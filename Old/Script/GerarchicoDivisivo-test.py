import numpy as np
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from Testing.Dataset import random_points_generator as rpg
from Testing.Clustering.PathetumEnviroment import ClusterEnvironment


class DivisiveClustering:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters

    def fit_predict(self, data):
        """
        Perform divisive clustering and return cluster labels.
        """
        # Initialize all points in one cluster
        clusters = {0: data.index.tolist()}
        current_cluster_count = 1

        # Distance matrix
        distances = squareform(pdist(data[['x', 'y']], metric='euclidean'))

        while current_cluster_count < self.n_clusters:
            # Find the cluster with the highest average dissimilarity
            cluster_to_split = max(clusters.keys(),
                                   key=lambda c: self._average_dissimilarity(clusters[c], distances))

            # Split the cluster into two
            cluster_indices = clusters.pop(cluster_to_split)
            new_labels = self._split_cluster(data.iloc[cluster_indices], distances[cluster_indices][:, cluster_indices])

            # Assign the split clusters to new clusters
            clusters[current_cluster_count] = [cluster_indices[i] for i in range(len(cluster_indices)) if new_labels[i] == 0]
            clusters[current_cluster_count + 1] = [cluster_indices[i] for i in range(len(cluster_indices)) if new_labels[i] == 1]
            current_cluster_count += 2

        # Create cluster labels
        labels = np.full(len(data), -1)
        for cluster_label, indices in clusters.items():
            labels[indices] = cluster_label

        return labels

    def _average_dissimilarity(self, indices, distances):
        """
        Calculate the average dissimilarity for a cluster.
        """
        cluster_distances = distances[np.ix_(indices, indices)]
        return np.mean(cluster_distances)

    def _split_cluster(self, data, distances):
        """
        Split a cluster into two using the farthest point from the mean as a seed.
        """
        # Find the farthest point from the cluster centroid
        centroid = data.mean().values
        farthest_point = np.argmax(np.linalg.norm(data[['x', 'y']].values - centroid, axis=1))

        # Initialize two clusters with the farthest point and the next farthest
        cluster1, cluster2 = [farthest_point], []
        for i in range(len(data)):
            if i != farthest_point:
                if distances[farthest_point, i] < np.mean(distances[farthest_point, :]):
                    cluster1.append(i)
                else:
                    cluster2.append(i)

        # Assign the split
        labels = np.zeros(len(data))
        labels[cluster2] = 1
        return labels


class ClusterAnalysis:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.cluster_env = ClusterEnvironment()

    def perform_clustering(self, data, n_clusters=None):
        """
        Perform divisive hierarchical clustering and visualize the results.
        """
        if n_clusters is None:
            n_clusters = self.n_clusters

        divisive_clustering = DivisiveClustering(n_clusters=n_clusters)
        data['label_cluster'] = divisive_clustering.fit_predict(data)

        # Calculate silhouette score
        if len(set(data['label_cluster'])) > 1:
            silhouette_avg = silhouette_score(data[['x', 'y']], data['label_cluster'])
            print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg:.2f}")
        else:
            silhouette_avg = None
            print("Silhouette Score: Not available (less than 2 clusters)")

        # Visualize results using ClusterEnvironment
        self.cluster_env.update_environment(data, n_clusters=n_clusters, step_title=f"Divisive Clustering - {n_clusters} clusters")

    def test_with_datasets(self):
        """
        Generate and test clustering on different datasets.
        """
        datasets = {
            "Uniform Data": rpg.generate_uniform_data(n_points=100, x_range=(0, 100), y_range=(0, 100)),
            "Gaussian Clusters": rpg.generate_gaussian_clusters(n_clusters=3, n_points_per_cluster=100, cluster_spread=5),
            "Overlapping Clusters": rpg.generate_overlapping_clusters(n_clusters=3, n_points_per_cluster=100, overlap=10),
            "Non-Spherical Clusters": rpg.generate_non_spherical_clusters(n_clusters=3, n_points_per_cluster=100, elongation=10),
            "Clusters with Outliers": rpg.generate_clusters_with_outliers(n_clusters=3, n_points_per_cluster=100, n_outliers=20)
        }

        for dataset_name, dataset in datasets.items():
            print(f"\nTesting {dataset_name}...")
            self.perform_clustering(dataset, n_clusters=self.n_clusters)


# Example usage
if __name__ == "__main__":
    cluster_analysis = ClusterAnalysis(n_clusters=3)
    cluster_analysis.test_with_datasets()

