from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
from Testing.Dataset import random_points_generator as rpg
from shapely.geometry import MultiPoint
import plotly.graph_objects as go



class ClusterEnvironment:
    def __init__(self):
        self.data = pd.DataFrame(columns=["x", "y", "label_cluster"])

    def update_environment(self, new_data, step_title="Algorithm Step"):
        """
        Update the environment with new data and visualize the results.
        """
        if isinstance(new_data, list):
            new_data = pd.DataFrame(new_data, columns=["x", "y"])
        if not {"x", "y"}.issubset(new_data.columns):
            raise ValueError("Input data must include 'x' and 'y' columns.")

        self.data = new_data.copy()
        self._visualize(step_title)

    def _visualize(self, title="Clustered Data"):
        """
        Visualize points and clusters using Plotly, including cluster polygons
        and colored points.
        """
        cluster_polygons = []
        unique_clusters = self.data["label_cluster"].unique()

        for cluster_label in unique_clusters:
            if cluster_label == -1:  # Skip noise points for convex hull
                continue

            cluster_points = self.data[self.data["label_cluster"] == cluster_label][["x", "y"]]

            # Create convex hull for cluster points
            multipoint = MultiPoint(cluster_points.values)
            convex_hull = multipoint.convex_hull

            if convex_hull.geom_type == 'Polygon':
                hull_coords = list(convex_hull.exterior.coords)
            elif convex_hull.geom_type == 'LineString':
                hull_coords = list(convex_hull.coords)
            else:
                continue

            hull_x = [coord[0] for coord in hull_coords]
            hull_y = [coord[1] for coord in hull_coords]

            cluster_polygons.append(go.Scatter(
                x=hull_x,
                y=hull_y,
                fill="toself",
                fillcolor=f"rgba({np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {np.random.randint(0, 255)}, 0.3)",
                line=dict(width=2, color='rgba(0,0,0,0)'),
                name=f"Cluster {cluster_label}"
            ))

        scatter = go.Scatter(
            x=self.data["x"],
            y=self.data["y"],
            mode="markers",
            marker=dict(
                size=10,
                color=self.data["label_cluster"],
                colorscale="Jet",
                opacity=0.6
            ),
            text=self.data["label_cluster"],
            hoverinfo="text+x+y"
        )

        layout = go.Layout(
            title=title,
            xaxis=dict(title="X", range=[0, 100]),
            yaxis=dict(title="Y", range=[0, 100]),
            showlegend=True
        )

        fig = go.Figure(data=[scatter] + cluster_polygons, layout=layout)
        fig.show()


class DBSCANAnalysis:
    def __init__(self, eps=5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.cluster_env = ClusterEnvironment()

    def perform_clustering(self, data):
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
        self.cluster_env.update_environment(data,
                                            step_title=f"DBSCAN Clustering (eps={self.eps}, min_samples={self.min_samples})")

        return dbscan, silhouette_avg

    def test_with_datasets(self):

        datasets = {
            "Uniform Data": rpg.generate_uniform_data(n_points=100, x_range=(0, 100), y_range=(0, 100)),
            "Gaussian Clusters": rpg.generate_gaussian_clusters(n_clusters=3, n_points_per_cluster=100, cluster_spread=5),
            "Overlapping Clusters": rpg.generate_overlapping_clusters(n_clusters=3, n_points_per_cluster=100, overlap=10),
            "Non-Spherical Clusters": rpg.generate_non_spherical_clusters(n_clusters=3, n_points_per_cluster=100, elongation=10),
            "Clusters with Outliers": rpg.generate_clusters_with_outliers(n_clusters=3, n_points_per_cluster=100, n_outliers=20)
        }

        for dataset_name, dataset in datasets.items():
            print(f"\nTesting {dataset_name}...")
            self.perform_clustering(dataset)


# Example usage
if __name__ == "__main__":
    # Example 1: Using default parameters for DBSCAN
    dbscan_analysis = DBSCANAnalysis(eps=5, min_samples=5)
    dbscan_analysis.test_with_datasets()

    # Example 2: Experiment with different DBSCAN parameters
    dbscan_analysis = DBSCANAnalysis(eps=10, min_samples=3)
    dbscan_analysis.test_with_datasets()
