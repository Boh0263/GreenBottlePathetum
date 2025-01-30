import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.figure_factory as ff
from Testing.Clustering.PathetumEnviroment import ClusterEnvironment
from Testing.Clustering.ClusterAnalysis import ClusterAnalysis


class HAClusterAnalysis(ClusterAnalysis):

    def __init__(self, linkage_method='ward', cluster_env=None):
        super().__init__()
        self.linkage_method = linkage_method  # 'ward', 'complete', 'average', 'single', or 'centroid'
        self.cluster_env = cluster_env if cluster_env else ClusterEnvironment()

    def run_hierarchical(self, data, distance_threshold=None, n_clusters=None):
        """
        Run Agglomerative Clustering with either a distance threshold or a specified number of clusters.
        """
        # Validate the parameters
        if distance_threshold is not None and n_clusters is not None:
            raise ValueError("Exactly one of 'distance_threshold' or 'n_clusters' must be provided, not both.")
        if distance_threshold is None and n_clusters is None:
            # Fallback to calculated best distance threshold
            distance_threshold = self.calculate_best_distance_threshold(data)
            print(f"Using calculated distance threshold: {distance_threshold}")

        # Initialize Agglomerative Clustering with the specified parameters
        if distance_threshold is not None:
            agglomerative = AgglomerativeClustering(
                n_clusters=None, distance_threshold=distance_threshold, linkage=self.linkage_method
            )
        else:
            agglomerative = AgglomerativeClustering(
                n_clusters=n_clusters, distance_threshold=None, linkage=self.linkage_method
            )

        # Fit and predict cluster labels
        data['label_cluster'] = agglomerative.fit_predict(data[['x', 'y']])
        return agglomerative, data

    def perform_clustering(self, data, distance_threshold=None, n_clusters=None):
        """
        Perform clustering using Agglomerative Clustering. Either a distance threshold
        or a number of clusters can be specified.
        """
        # Run the hierarchical clustering
        model, clustered_data = self.run_hierarchical(data, distance_threshold=distance_threshold, n_clusters=n_clusters)

        # Visualize results using ClusterEnvironment
        n_clusters_result = len(np.unique(clustered_data['label_cluster']))
        self.cluster_env.update_environment(
            clustered_data, n_clusters=n_clusters_result,
            step_title=f"Agglomerative Clustering - {n_clusters_result} clusters"
        )

        return model

    def calculate_best_distance_threshold(self, data):
        """
        Calculate the best distance threshold based on the largest gap in the linkage matrix.
        """
        # Compute the linkage matrix
        linked = linkage(data[['x', 'y']], method=self.linkage_method)

        # Extract distances from the linkage matrix
        distances = linked[:, 2]  # Column 2 contains the distances of merges
        sorted_distances = np.sort(distances)

        # Calculate the differences between successive distances
        distance_gaps = np.diff(sorted_distances)

        # Find the largest gap and determine the best threshold
        largest_gap_index = np.argmax(distance_gaps)
        best_threshold = sorted_distances[largest_gap_index + 1]  # Threshold just after the largest gap

        return best_threshold

    def plot_dendrogram(self, data, threshold_suggestion=None):
        """
        Plot an interactive dendrogram for hierarchical clustering using Plotly.
        """
        # Compute linkage matrix
        linked = linkage(data[['x', 'y']], method=self.linkage_method)

        # Create Plotly dendrogram
        fig = ff.create_dendrogram(
            linked,
            orientation='bottom',
            color_threshold=threshold_suggestion
        )

        # Retrieve leaf node positions (the individual data points)
        leaf_positions = np.array([i for i in range(len(data))])

        # Generate the number of ticks for the X-axis (all data points)
        labels = list(data.index)  # Default to using data indices as labels
        label_interval = 10  # Show every 10th label (or choose a different number)

        sampled_labels = labels[::label_interval]  # Select every N-th label
        tickvals = leaf_positions[::label_interval]  # Corresponding tick positions

        fig.update_layout(
            xaxis=dict(
                tickvals=tickvals,
                ticktext=sampled_labels,
                tickangle=90,  # Rotate labels to prevent overlap
                tickfont=dict(size=10)  # Adjust font size
            )
        )

        # Add a red horizontal line to indicate the suggested threshold (if provided)
        if threshold_suggestion:
            # Ensure the line spans the entire X-axis
            fig.add_shape(
                type="line",
                x0=0,
                x1=len(data) - 1,  # Extend the threshold line across all data points
                y0=threshold_suggestion,
                y1=threshold_suggestion,
                line=dict(color="red", width=2, dash="dash"),
                xref="x",
                yref="y"
            )

        # Increase the figure width and height to provide more space for labels and tree
        fig.update_layout(
            title=f"Dendrogram (Linkage: {self.linkage_method.capitalize()})",
            xaxis_title="Data Points",
            yaxis_title="Distance",
            showlegend=True,
            template="plotly_white",
            width=2000,  # Increase width
            height=800,  # Adjust height
            margin=dict(l=50, r=50, t=50, b=150)  # Ensure enough bottom margin for labels
        )

        # Display interactive dendrogram
        fig.show()

