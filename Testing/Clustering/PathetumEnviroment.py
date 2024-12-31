import pandas as pd
import numpy as np
from shapely.geometry import MultiPoint
import plotly.graph_objects as go


class ClusterEnvironment:
    def __init__(self):
        self.data = pd.DataFrame(columns=["x", "y", "label_cluster"])

    def update_environment(self, new_data, n_clusters=3, step_title="Algorithm Step"):
        """
        Update the environment with new data and visualize the results.
        """
        if isinstance(new_data, list):
            new_data = pd.DataFrame(new_data, columns=["x", "y"])
        if not {"x", "y"}.issubset(new_data.columns):
            raise ValueError("Input data must include 'x' and 'y' columns.")

        # Update the data environment
        self.data = new_data.copy()
        self._visualize(step_title)

    def _visualize(self, title="Clustered Data"):

        cluster_polygons = []
        for cluster_label in self.data["label_cluster"].unique():
            if cluster_label == -1:
                # Skip outliers for polygons
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

            # Use a more appealing color scheme
            cluster_color = f"rgba({np.random.randint(50, 255)}, {np.random.randint(50, 255)}, {np.random.randint(50, 255)}, 0.3)"

            cluster_polygons.append(go.Scatter(
                x=hull_x,
                y=hull_y,
                fill="toself",
                fillcolor=cluster_color,
                line=dict(width=2, color='rgba(0,0,0,0)'),
                name=f"Cluster {cluster_label}",
                hoverinfo="text"
            ))

        # Plot the points with a glowing effect and more vibrant colors
        scatter = go.Scatter(
            x=self.data["x"],
            y=self.data["y"],
            mode="markers",
            marker=dict(
                size=12,
                color=self.data["label_cluster"],
                colorscale="Rainbow",  # Vibrant rainbow colors
                opacity=0.8,  # Increase opacity for better visibility
                line=dict(width=2, color='black'),  # Black outline for points to make them pop
                showscale=False  # Hide the color scale (slider)
            ),
            text=self.data["label_cluster"].apply(lambda x: "Outliers" if x == -1 else f"Cluster {x}"),
            hoverinfo="text+x+y",
            name="Data Points"
        )

        # Set a cool gradient background and make it visually striking
        layout = go.Layout(
            title=title,
            title_font=dict(size=24, family='Arial, sans-serif', color='black'),
            xaxis=dict(title="X", range=[0, 100], showgrid=False, zeroline=False),
            yaxis=dict(title="Y", range=[0, 100], showgrid=False, zeroline=False),
            showlegend=True,

            font=dict(color='black'),
            hoverlabel=dict(bgcolor='white', font=dict(color='black')),
            shapes=[{
                'type': 'rect',
                'x0': 0,
                'y0': 0,
                'x1': 1,
                'y1': 1,
                'xref': 'paper',
                'yref': 'paper',
                'fillcolor': 'rgba(0, 0, 0, 0.1)',
                'layer': 'below',
                'line': {'width': 0}
            }],
            xaxis_showgrid=True,
            yaxis_showgrid=True
        )

        # Create the figure and show it
        fig = go.Figure(data=[scatter] + cluster_polygons, layout=layout)
        fig.show()
