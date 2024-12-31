from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from Testing.Clustering.PathetumEnviroment import ClusterEnvironment
from Testing.Clustering.ClusterAnalysis import ClusterAnalysis


class DBSCANAnalysis(ClusterAnalysis):
    def __init__(self, eps=5, min_samples=5):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.cluster_env = ClusterEnvironment()

    def perform_clustering(self, data, **kwargs):
        """
        Perform DBSCAN clustering and visualize the results.
        """
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        data['label_cluster'] = dbscan.fit_predict(data[['x', 'y']])

        # Check the number of unique clusters (ignoring noise points, i.e., label -1)
        unique_clusters = set(data['label_cluster'])
        if len(unique_clusters - {-1}) >= 2:  # At least 2 clusters (excluding noise)
            silhouette_avg = silhouette_score(
                data[data['label_cluster'] != -1][['x', 'y']],
                data[data['label_cluster'] != -1]['label_cluster']
            )
            print(f"Silhouette Score: {silhouette_avg:.2f}")
        else:
            silhouette_avg = None
            print("Silhouette Score: Not available (less than 2 clusters)")

        # Visualize results using ClusterEnvironment
        self.cluster_env.update_environment(data, step_title=f"DBSCAN Clustering (eps={self.eps}, min_samples={self.min_samples})")

        return dbscan, silhouette_avg