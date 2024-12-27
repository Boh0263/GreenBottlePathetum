import pandas as pd
import numpy as np
import plotly.graph_objects as go
from shapely.geometry import MultiPoint


class ClusterEnvironment:
    def __init__(self):
        self.data = pd.DataFrame(columns=["x", "y", "label_cluster"])

    def update_environment(self, new_data, step_title="Algorithm Step"):

        if isinstance(new_data, list):
            new_data = pd.DataFrame(new_data)
        if not {"x", "y"}.issubset(new_data.columns):
            raise ValueError("Input data must include 'x' and 'y' columns.")

        self.data = new_data.copy()
        self._visualize(step_title)

    def _visualize(self, title="Clustered Data"):
        cluster_polygons = []
        for cluster_label in self.data["label_cluster"].unique():

            cluster_points = self.data[self.data["label_cluster"] == cluster_label][["x", "y"]]

            multipoint = MultiPoint(cluster_points.values)
            convex_hull = multipoint.convex_hull

            # Handle if the convex hull is a Polygon or LineString
            if convex_hull.geom_type == 'Polygon':
                # For Polygon, extract coordinates from exterior
                hull_coords = list(convex_hull.exterior.coords)
            elif convex_hull.geom_type == 'LineString':
                # For LineString, extract coordinates directly
                hull_coords = list(convex_hull.coords)
            else:
                # Handle unexpected geometry types
                print(f"Unexpected geometry type: {convex_hull.geom_type}")
                continue

            hull_x = [coord[0] for coord in hull_coords]
            hull_y = [coord[1] for coord in hull_coords]

            # Add the polygon for the cluster
            cluster_polygons.append(go.Scatter(
                x=hull_x,
                y=hull_y,
                fill="toself",  # Fill the cluster area
                fillcolor=f"rgba({np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {np.random.randint(0, 255)}, 0.3)",
                # Random color with transparency
                line=dict(width=2, color='rgba(0,0,0,0)'),  # No boundary line
                name=f"Cluster {cluster_label}"
            ))

        # Create a Plotly scatter plot for the points
        scatter = go.Scatter(
            x=self.data["x"],
            y=self.data["y"],
            mode="markers",
            marker=dict(
                size=10,
                color=self.data["label_cluster"],  # Set color based on cluster_label
                colorscale="Jet",  # Customize color scale
                opacity=0.6  # Set transparency for the entire scatter plot (0 = fully transparent, 1 = opaque)
            ),
            text=self.data["label_cluster"],  # Add cluster label to hover info
            hoverinfo="text+x+y"  # Show x, y, and cluster label on hover
        )

        # Set up the layout for the Cartesian plot
        layout = go.Layout(
            title=title,
            xaxis=dict(title="X", range=[0, 100]),  # X-axis range
            yaxis=dict(title="Y", range=[0, 100]),  # Y-axis range
            showlegend=True
        )

        # Create the figure and show it
        fig = go.Figure(data=[scatter] + cluster_polygons, layout=layout)
        fig.show()