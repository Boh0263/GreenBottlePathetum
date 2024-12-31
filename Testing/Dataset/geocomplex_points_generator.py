import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import random

"""

Questo script genera punti che rappresentano la distribuzione dei clienti in base a tre fattori principali:

1. Densità radiale: I punti sono più concentrati vicino al centro città e la densità diminuisce con la distanza, utilizzando il centro geometrico di ciascuna città (dallo shapefile delle municipalità).
2. Prossimità alle strade: Una parte dei punti è generata vicino alle reti stradali, utilizzando un buffer definito attorno alle geometrie delle strade (dallo shapefile delle strade).
3. Distribuzione rurale: I punti sono distribuiti casualmente nelle aree rurali, evitando le zone ad alta altitudine, basandosi su una mappa altimetrica (array 2D di altitudine) e sugli estremi geografici forniti dagli shapefile.

Lo script utilizza gli shapefile delle municipalità per le forme delle città e delle strade per determinare la posizione dei punti.
Inoltre, si considera una mappa altimetrica per evitare zone ad alta altitudine,
influenzando la posizione dei punti in base alla loro elevazione.

"""

def generate_points_near_roads(road_geodata, n_points, buffer_distance=1):
    """
    Genera punti casuali vicino alle reti stradali.
    """
    road_buffer = road_geodata.buffer(buffer_distance)
    combined_buffer = road_buffer.unary_union  # Merge all buffers
    points = []
    minx, miny, maxx, maxy = road_geodata.total_bounds

    while len(points) < n_points:
        random_point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if combined_buffer.contains(random_point):
            points.append({'x': random_point.x, 'y': random_point.y})
    return points


def generate_points_with_density_gradient(city_center, num_points, max_distance, altitude_map, altitude_threshold=70):
    """
    Genera punti con un gradiente di densità radiale, favorendo le aree più vicine al centro città.
    """
    points = []
    center_x, center_y = city_center
    while len(points) < num_points:
        angle = random.uniform(0, 2 * np.pi)
        distance = random.uniform(0, max_distance)
        x = center_x + distance * np.cos(angle)
        y = center_y + distance * np.sin(angle)
        altitude = altitude_map[int(y), int(x)]
        if altitude <= altitude_threshold:
            density_factor = 1 - (distance / max_distance)  # Higher density closer to the center
            if random.random() < density_factor:  # Probability drops with distance
                points.append({'x': x, 'y': y})
    return points


def generate_city_and_scattered_customers_complex(shapefile_path, population_data, road_shapefile, altitude_map, altitude_threshold=70):
    """
    Genera punti clienti basati sulle forme della città, le strade e una distribuzione rurale sparsa.
    """
    city_boundaries = gpd.read_file(shapefile_path)
    road_geodata = gpd.read_file(road_shapefile)
    map_bounds = city_boundaries.total_bounds  # [min_x, min_y, max_x, max_y]
    all_points = []

    # Generate points for each city
    for _, city in city_boundaries.iterrows():
        city_name = city['name']  # Adjust field name
        city_population = population_data.get(city_name, 0)

        if city_population > 0:
            num_points = int(city_population / 1000)  # Scale population for manageability
            city_center = city.geometry.centroid.coords[0]
            max_distance = 10  # Adjust based on city size

            city_points = generate_points_with_density_gradient(
                city_center, num_points, max_distance, altitude_map, altitude_threshold
            )
            all_points.extend(city_points)

            road_points = generate_points_near_roads(road_geodata, int(num_points * 0.1))  # 10% near roads
            all_points.extend(road_points)

    n_rural_points = int(sum(population_data.values()) * 0.3 / 1000)  # 30% della popolazione in aree rurali
    rural_points = generate_random_points_on_map(n_rural_points, map_bounds, altitude_map, altitude_threshold)
    all_points.extend(rural_points)

    return pd.DataFrame(all_points)


def generate_random_points_on_map(n_points, map_bounds, altitude_map, altitude_threshold=70, high_altitude_penalty=0.1):
    """
    Genera punti random sulla mappa, evitando frequentemente zone alte.
    """
    points = []
    while len(points) < n_points:
        x = random.uniform(map_bounds[0], map_bounds[2])  # min_x, max_x
        y = random.uniform(map_bounds[1], map_bounds[3])  # min_y, max_y
        altitude = altitude_map[int(y), int(x)]

        # Penalizza zone alte.
        if altitude < altitude_threshold or random.random() < high_altitude_penalty:
            points.append({'x': x, 'y': y})
    return points

shapefile_path = 'path_to_shapefile/italy_municipalities.shp' #Dati recuperati da OpenStreetMap (richiede menzione su github) Work in progress
road_shapefile = 'path_to_shapefile/italy_roads.shp' #Dati recuperati da OpenStreetMap (richiede menzione su github) Work in progress
population_data = {
    'Rome': 2750000,
    'Milan': 1378689,
    'Naples': 962589,
    #Dati recuperati dalla banca dati dell' ISTAT.
    #Work In progress
}
altitude_map = np.random.normal(50, 15, (100, 100))  # Replace with actual altitude data

customer_data = generate_city_and_scattered_customers_complex(
    shapefile_path, population_data, road_shapefile, altitude_map
)
