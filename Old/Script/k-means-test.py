import pandas as pd
import numpy as np
from Testing.Dataset import random_points_generator as rpg
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint
import plotly.graph_objects as go


class ClusterEnvironment:
    def __init__(self):
        self.data = pd.DataFrame(columns=["x", "y", "label_cluster"])

    def update_environment(self, new_data, n_clusters=3, step_title="Algorithm Step"):
        """
        Update the environment with new data, calculate KMeans clusters,
        and visualize the results.
        """
        if isinstance(new_data, list):
            new_data = pd.DataFrame(new_data, columns=["x", "y"])
        if not {"x", "y"}.issubset(new_data.columns):
            raise ValueError("Input data must include 'x' and 'y' columns.")

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        new_data['label_cluster'] = kmeans.fit_predict(new_data[['x', 'y']])

        # Update the data environment
        self.data = new_data.copy()
        self._visualize(step_title)

    def _visualize(self, title="Clustered Data"):
        """
        Visualize points and clusters using Plotly, including cluster polygons
        and colored points.
        """
        cluster_polygons = []
        for cluster_label in self.data["label_cluster"].unique():
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


class ClusterAnalysis:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.cluster_env = ClusterEnvironment()

    def perform_clustering(self, data, n_clusters=None):
        """
        Perform KMeans clustering and visualize the results.
        """
        if n_clusters is None:
            n_clusters = self.n_clusters

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        data['label_cluster'] = kmeans.fit_predict(data[['x', 'y']])

        # Calculate silhouette score
        silhouette_avg = silhouette_score(data[['x', 'y']], data['label_cluster'])
        print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg:.2f}")

        # Visualize results using ClusterEnvironment
        self.cluster_env.update_environment(data, n_clusters=n_clusters, step_title=f"KMeans Clustering - {n_clusters} clusters")

        return kmeans, silhouette_avg

    def elbow_method(self, data, max_clusters=10):
        """
        Use the Elbow Method to find the optimal number of clusters.
        """
        inertia_values = []

        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(data[['x', 'y']])
            inertia_values.append(kmeans.inertia_)

        plt.figure(figsize=(10, 6))
        plt.plot(range(2, max_clusters + 1), inertia_values, marker='o', label="Inertia")
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.legend()
        plt.show()

        inertia_diff = np.diff(inertia_values)
        inertia_diff2 = np.diff(inertia_diff)
        elbow_index = np.argmax(inertia_diff2) + 2

        print(f"Optimal number of clusters (Elbow Method): {elbow_index}")
        return elbow_index

    def silhouette_method(self, data, max_clusters=10):
        """
        Use the Silhouette Method to find the optimal number of clusters.
        """
        silhouette_scores = []

        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(data[['x', 'y']])
            silhouette_avg = silhouette_score(data[['x', 'y']], labels)
            silhouette_scores.append(silhouette_avg)

        plt.figure(figsize=(10, 6))
        plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o', label="Silhouette Score")
        plt.title('Silhouette Method')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.legend()
        plt.show()

        optimal_k = np.argmax(silhouette_scores) + 2
        print(f"Optimal number of clusters (Silhouette Method): {optimal_k}")
        return optimal_k

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

            # Elbow Method
            optimal_k_elbow = self.elbow_method(dataset, max_clusters=10)
            print(f"Optimal number of clusters (Elbow Method) for {dataset_name}: {optimal_k_elbow}")

            # Silhouette Method
            optimal_k_silhouette = self.silhouette_method(dataset, max_clusters=10)
            print(f"Optimal number of clusters (Silhouette Method) for {dataset_name}: {optimal_k_silhouette}")

            # Perform clustering with the optimal k from the elbow method
            print(f"\nPerforming clustering for {dataset_name} with k={optimal_k_elbow} (Elbow Method)...")
            self.perform_clustering(dataset, n_clusters=optimal_k_elbow)


# Example usage
if __name__ == "__main__":
    cluster_analysis = ClusterAnalysis()
    cluster_analysis.test_with_datasets()
