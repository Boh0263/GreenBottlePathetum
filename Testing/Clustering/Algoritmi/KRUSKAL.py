import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from Testing.Clustering.PathetumEnviroment import ClusterEnvironment
from Testing.Clustering.ClusterAnalysis import ClusterAnalysis
from Testing.Clustering.Algoritmi.KMEANS import KMeansAnalysis


class KruskalClustering(ClusterAnalysis):

    def __init__(self, n_clusters=3, max_clusters=10, cluster_env=None):
        super().__init__()
        self.n_clusters = n_clusters  # Maximum allowed clusters
        self.max_clusters = max_clusters
        self.cluster_env = cluster_env if cluster_env else ClusterEnvironment()

    def run_kruskal(self, data, n_clusters):
        """
        Perform Kruskal-based clustering.
        """
        from scipy.sparse.csgraph import connected_components

        # Compute pairwise distance matrix
        dist_matrix = distance_matrix(data[['x', 'y']], data[['x', 'y']])

        # Build the Minimum Spanning Tree (MST)
        mst = minimum_spanning_tree(dist_matrix).toarray()

        # Get the sorted list of edges (distance, i, j)
        edges = sorted(
            [(mst[i, j], i, j) for i in range(len(mst)) for j in range(len(mst)) if mst[i, j] > 0],
            key=lambda x: x[0],
            reverse=True
        )

        # Remove the longest n_clusters - 1 edges
        for _ in range(n_clusters - 1):
            if edges:
                _, i, j = edges.pop(0)
                mst[i, j] = 0
                mst[j, i] = 0  # Ensure symmetry

        # Assign clusters using connected components
        n_components, labels = connected_components(csgraph=mst, directed=False)

        data['label_cluster'] = labels
        return data, n_components

    def perform_clustering(self, data):
        """
        Determine the optimal number of clusters using KMeans silhouette method,
        and perform Kruskal clustering with a cap on the maximum number of clusters.
        """
        # Use KMeans silhouette method to find the optimal number of clusters
        kmeans_analysis = KMeansAnalysis(max_clusters=self.max_clusters)
        optimal_clusters = kmeans_analysis.silhouette_method(data)

        print(f"Optimal number of clusters (Silhouette Method): {optimal_clusters}")

        # Cap the number of clusters at the maximum allowed value
        capped_clusters = min(optimal_clusters, self.n_clusters)

        print(f"Using {capped_clusters} clusters for Kruskal's Clustering (Capped at {self.n_clusters}).")

        # Perform Kruskal clustering
        clustered_data, n_components = self.run_kruskal(data, capped_clusters)

        print(f"Kruskal's Clustering formed {n_components} clusters.")

        # Visualize results using ClusterEnvironment
        self.cluster_env.update_environment(clustered_data, n_clusters=n_components,
                                            step_title=f"Kruskal's Clustering - {n_components} clusters")

        return clustered_data

    def visualize_mst(self, data):
        """
        Visualize the Minimum Spanning Tree (MST).
        """
        # Compute pairwise distance matrix
        dist_matrix = distance_matrix(data[['x', 'y']], data[['x', 'y']])

        # Build the MST
        mst = minimum_spanning_tree(dist_matrix).toarray()

        # Plot the points
        plt.figure(figsize=(10, 6))
        plt.scatter(data['x'], data['y'], c='blue', label="Data Points")
        plt.title("Minimum Spanning Tree (MST)")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")

        # Plot the edges of the MST
        for i in range(len(mst)):
            for j in range(len(mst)):
                if mst[i, j] > 0:
                    plt.plot([data.iloc[i]['x'], data.iloc[j]['x']],
                             [data.iloc[i]['y'], data.iloc[j]['y']],
                             color='red', linestyle='--')

        plt.legend()
        plt.show()