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
        Update the environment with new data, calculate KMeans clusters,
        and visualize the results.
        """
        if isinstance(new_data, list):
            new_data = pd.DataFrame(new_data)
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

            # Filter points for the current cluster
            cluster_points = self.data[self.data["label_cluster"] == cluster_label][["x", "y"]]

            # Create convex hull for cluster points
            multipoint = MultiPoint(cluster_points.values)
            convex_hull = multipoint.convex_hull

            # Handle convex hull geometry types
            if convex_hull.geom_type == 'Polygon':
                hull_coords = list(convex_hull.exterior.coords)
            elif convex_hull.geom_type == 'LineString':
                hull_coords = list(convex_hull.coords)
            else:
                continue

            # Separate x and y coordinates of the hull
            hull_x = [coord[0] for coord in hull_coords]
            hull_y = [coord[1] for coord in hull_coords]

            # Add the cluster polygon
            cluster_polygons.append(go.Scatter(
                x=hull_x,
                y=hull_y,
                fill="toself",
                fillcolor=f"rgba({np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {np.random.randint(0, 255)}, 0.3)",
                line=dict(width=2, color='rgba(0,0,0,0)'),
                name=f"Cluster {cluster_label}"
            ))

        # Create scatter plot for the points
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

        # Set layout
        layout = go.Layout(
            title=title,
            xaxis=dict(title="X", range=[0, 100]),
            yaxis=dict(title="Y", range=[0, 100]),
            showlegend=True
        )

        # Create and show the figure
        fig = go.Figure(data=[scatter] + cluster_polygons, layout=layout)
        fig.show()


class ClusterAnalysis:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.cluster_env = ClusterEnvironment()

    def perform_clustering(self, data, n_clusters=None):
        """
        Perform clustering with KMeans and update the environment for each step.
        """
        if n_clusters is None:
            n_clusters = self.n_clusters

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data[['x', 'y']])

        # Add cluster labels to the data
        data['label_cluster'] = labels

        # Calculate silhouette score
        silhouette_avg = silhouette_score(data[['x', 'y']], labels)
        print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg:.2f}")

        # Visualize the clustering result
        self.cluster_env.update_environment(data, n_clusters=n_clusters,
                                            step_title=f"KMeans Clustering - {n_clusters} clusters")
        return labels, kmeans, silhouette_avg

    def elbow_method(self, data, max_clusters=10):
        """
        Elbow method to find the optimal number of clusters.
        """
        inertia_values = []

        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(data[['x', 'y']])
            inertia_values.append(kmeans.inertia_)

        # Plot inertia values
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, max_clusters + 1), inertia_values, marker='o', label="Inertia")
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.legend()
        plt.grid()
        plt.show()

        # Find the elbow point (maximum curvature)
        diff = np.diff(inertia_values)
        diff2 = np.diff(diff)
        elbow_index = np.argmax(diff2) + 2  # Offset for cluster index

        print(f"Optimal number of clusters based on elbow method: {elbow_index}")
        return elbow_index

    def silhouette_method(self, data, max_clusters=10):
        """
        Silhouette method to find the optimal number of clusters.
        """
        silhouette_scores = []

        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(data[['x', 'y']])
            score = silhouette_score(data[['x', 'y']], labels)
            silhouette_scores.append(score)

        # Plot silhouette scores
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o', label="Silhouette Score")
        plt.title('Silhouette Method')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.legend()
        plt.grid()
        plt.show()

        # Find the number of clusters with the highest silhouette score
        optimal_clusters = np.argmax(silhouette_scores) + 2  # Offset for cluster index

        print(f"Optimal number of clusters based on silhouette method: {optimal_clusters}")
        return optimal_clusters


# Main Program
if __name__ == "__main__":
    # Generate random dataset
    np.random.seed(42)
    n_points = 100
    new_data = np.random.rand(n_points, 2) * 100  # 100 random points within 0-100 range
    df_new_data = pd.DataFrame(new_data, columns=["x", "y"])

    cluster_analysis = ClusterAnalysis()

    print("Performing Elbow Method...")
    optimal_clusters_elbow = cluster_analysis.elbow_method(df_new_data, max_clusters=10)

    print("Performing Silhouette Method...")
    optimal_clusters_silhouette = cluster_analysis.silhouette_method(df_new_data, max_clusters=10)

    print("Performing clustering with optimal number of clusters...")
    labels, kmeans_model, silhouette_avg = cluster_analysis.perform_clustering(df_new_data, n_clusters=optimal_clusters_elbow)

    print("Performing clustering with optimal number of clusters from Silhouette...")
    labels1, kmeans_model1, silhouette_avg1 = cluster_analysis.perform_clustering(df_new_data, n_clusters=optimal_clusters_silhouette)

    print(f"Final Silhouette Score with {optimal_clusters_elbow} clusters: {silhouette_avg:.2f}")
    print(f"Final Silhouette Score with {optimal_clusters_silhouette} clusters: {silhouette_avg1:.2f}")
