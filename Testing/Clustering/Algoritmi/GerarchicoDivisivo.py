import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from Testing.Clustering.PathetumEnviroment import ClusterEnvironment
from Testing.Clustering.ClusterAnalysis import ClusterAnalysis


class HDClusterAnalysis(ClusterAnalysis):
    def __init__(self, n_clusters=None):
        super().__init__()
        self.n_clusters = n_clusters  # This can be None, or an integer
        self.cluster_env = ClusterEnvironment()

    def fit_predict(self, data):
        """
        Esegue il clustering gerarchico divisivo e restituisci le etichette dei cluster.
        Se è specificato un numero di cluster (n_clusters), l'algoritmo si ferma quando si raggiunge quel numero.
        Se n_clusters è impostato su None, l'algoritmo si ferma quando viene raggiunta una condizione di arresto naturale.
        """
        # Inizializza tutti i punti in un unico cluster.
        clusters = {0: data.index.tolist()}
        current_cluster_count = 1

        # Matrice delle distanze
        distances = squareform(pdist(data[['x', 'y']], metric='euclidean'))

      #Continua a suddividere fino a quando non si verifica una delle seguenti condizioni:
        #1. Il numero di cluster specificato (`n_clusters`) viene raggiunto (se è stato specificato).
        #2. Viene soddisfatta una condizione di arresto naturale (se `n_clusters` è impostato su None).

        while True:
            if self.n_clusters is not None and len(clusters) >= self.n_clusters:
                break

            # Find the cluster with the highest average dissimilarity
            cluster_to_split = max(clusters.keys(), key=lambda c: self._average_dissimilarity(clusters[c], distances))

            # Split the cluster into two
            cluster_indices = clusters.pop(cluster_to_split)
            new_labels = self._split_cluster(data.iloc[cluster_indices], distances[cluster_indices][:, cluster_indices])

            # Assign the split clusters to new clusters
            clusters[current_cluster_count] = [cluster_indices[i] for i in range(len(cluster_indices)) if new_labels[i] == 0]
            clusters[current_cluster_count + 1] = [cluster_indices[i] for i in range(len(cluster_indices)) if new_labels[i] == 1]
            current_cluster_count += 2

            # If n_clusters is None, check if we should stop based on natural conditions
            if self.n_clusters is None and not self._should_continue(clusters, distances):
                break

        # Crea label per i cluster
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

    def _should_continue(self, clusters, distances):
        """
        Determine whether the clustering process should continue.
        This is a stopping condition function.
        """
        # Example stopping condition 1: If all clusters have size 1 (can't split further)
        if any(len(indices) == 1 for indices in clusters.values()):
            return False

        # Example stopping condition 2: If the average dissimilarity between clusters is small
        # Compute average dissimilarity between clusters
        avg_dissimilarity = np.mean([self._average_dissimilarity(indices, distances) for indices in clusters.values()])
        if avg_dissimilarity < 0.1:  # Set a threshold value
            return False

        return True

    def perform_clustering(self, data):
        """
        Perform divisive hierarchical clustering and visualize the results.
        """
        labels = self.fit_predict(data)

        # Assign the generated labels to the data
        data['label_cluster'] = labels

        # Calculate silhouette score (if possible)
        if len(set(data['label_cluster'])) > 1:
            silhouette_avg = silhouette_score(data[['x', 'y']], data['label_cluster'])
            print(f"Silhouette Score: {silhouette_avg:.2f}")
        else:
            silhouette_avg = None
            print("Silhouette Score: Not available (less than 2 clusters)")

        # Visualize results using ClusterEnvironment
        self.cluster_env.update_environment(data, step_title="Divisive Clustering")

        # Return the data with labels
        return data
