import geopandas as gpd
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from shapely.geometry import MultiPoint


class ClusterEnvironment:
    def __init__(self, italy_shapefile=None):
        self.italy_map = None
        if italy_shapefile:
            self.italy_map = gpd.read_file(italy_shapefile)  # Load Italy map

        # Initialize an empty dataframe with required columns
        self.data = pd.DataFrame(columns=["latitudine", "longitudine", "label_cluster"])

    def update_environment(self, new_data, step_title="Algorithm Step"):
        # Convert new_data to a DataFrame if it's a list of dictionaries
        if isinstance(new_data, list):
            new_data = pd.DataFrame(new_data)

        # Validate required columns
        if not {"latitudine", "longitudine"}.issubset(new_data.columns):
            raise ValueError("Input data deve includere le colonne 'latitudine' e 'longitudine'.")

        # Update the data with new coordinates and cluster labels
        self.data = new_data.copy()

        # Visualize the updated environment
        self._visualize(step_title)

    def _visualize(self, title="Clustered Data"):
        # Extract coordinates of the Italy boundary
        italy_boundary_lons = []
        italy_boundary_lats = []

        for geom in self.italy_map.geometry.boundary:
            if geom.geom_type == 'Polygon':
                coords = list(geom.exterior.coords)
            elif geom.geom_type == 'MultiPolygon':
                coords = [coord for polygon in geom.geoms for coord in polygon.exterior.coords]
            else: continue

            italy_boundary_lons.extend([coord[0] for coord in coords])
            italy_boundary_lats.extend([coord[1] for coord in coords])

        # Create the Italy shapefile boundary using Scattermapbox
        italy_boundary = go.Scattermapbox(
            lat=italy_boundary_lats,
            lon=italy_boundary_lons,
            mode='lines',
            line=dict(width=2, color='black'),  # Set the boundary color (black)
            name="Italy Boundary"
        )

        # Create Convex Hulls for each cluster to represent areas
        cluster_polygons = []
        for cluster_label in self.data["label_cluster"].unique():
            # Get points for the current cluster
            cluster_points = self.data[self.data["label_cluster"] == cluster_label][["longitudine", "latitudine"]]
            # Create a MultiPoint object for the cluster
            multipoint = MultiPoint(cluster_points.values)
            # Generate the convex hull for the cluster
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

            hull_lons = [coord[0] for coord in hull_coords]
            hull_lats = [coord[1] for coord in hull_coords]

            # Add the polygon for the cluster
            cluster_polygons.append(go.Scattermapbox(
                lat=hull_lats,
                lon=hull_lons,
                fill="toself",  # Fill the cluster area
                fillcolor=f"rgba({np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {np.random.randint(0, 255)}, 0.3)",
                # Random color with transparency
                line=dict(width=2, color='rgba(0,0,0,0)'),  # No boundary line
                name=f"Cluster {cluster_label}"
            ))

        # Create a Plotly scatter plot with transparency for the points
        scatter = go.Scattermapbox(
            lat=self.data["latitudine"],
            lon=self.data["longitudine"],
            mode="markers",
            marker=dict(
                size=10,
                color=self.data["label_cluster"],  # Set color based on cluster_label
                colorscale="Jet",  # Customize color scale
                opacity=0.6  # Set transparency for the entire scatter plot (0 = fully transparent, 1 = opaque)
            ),
            text=self.data["label_cluster"],  # Add cluster label to hover info
            hoverinfo="text+lat+lon"  # Show lat, lon, and cluster label on hover
        )

        # Set up the layout for the map with a transparent background for the map
        layout = go.Layout(
            title=title,
            mapbox=dict(
                style="open-street-map",  # Use OpenStreetMap as the background
                center=dict(
                    lat=self.data["latitudine"].mean(),  # Center the map on the average lat/lon of the points
                    lon=self.data["longitudine"].mean()
                ),
                zoom=5,  # Adjust zoom level based on data range
            ),
            showlegend=True
        )

        # Create the figure and show it
        fig = go.Figure(data=[italy_boundary, scatter] + cluster_polygons, layout=layout)
        fig.show()