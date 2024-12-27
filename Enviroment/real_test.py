from PathetumEnviroment import ClusterEnvironment

def main():
    italy_shapefile_path = "italy_Italy_Country_Boundary.shp"
    env = ClusterEnvironment(italy_shapefile=italy_shapefile_path)

    # Step 0: Initial Data - Unset cluster labels (label = -1)
    initial_data = [
        {"latitudine": 45.4642, "longitudine": 9.1900, "label_cluster": -1},  # Milan
        {"latitudine": 41.9028, "longitudine": 12.4964, "label_cluster": -2},  # Rome
        {"latitudine": 40.8518, "longitudine": 14.2681, "label_cluster": -3},  # Naples
        {"latitudine": 39.2238, "longitudine": 9.1217, "label_cluster": -4},  # Cagliari
        {"latitudine": 44.4949, "longitudine": 11.3426, "label_cluster": -5},  # Bologna
        {"latitudine": 43.7696, "longitudine": 11.2558, "label_cluster": -6},  # Florence
        {"latitudine": 42.3601, "longitudine": 13.4315, "label_cluster": -7},  # Pescara
        {"latitudine": 45.0703, "longitudine": 7.6869, "label_cluster": -8},  # Turin
        {"latitudine": 46.6034, "longitudine": 9.0149, "label_cluster": -9},  # Sondrio
        {"latitudine": 38.1157, "longitudine": 13.3615, "label_cluster": -10},  # Palermo
        {"latitudine": 37.7749, "longitudine": 15.0858, "label_cluster": -11},  # Catania
        {"latitudine": 40.8522, "longitudine": 14.2681, "label_cluster": -12},  # Naples
        {"latitudine": 43.7167, "longitudine": 10.4000, "label_cluster": -13},  # Pisa
        {"latitudine": 42.0560, "longitudine": 11.8818, "label_cluster": -14},  # Viterbo
        {"latitudine": 45.0525, "longitudine": 7.6774, "label_cluster": -15},  # Asti
        {"latitudine": 38.0472, "longitudine": 15.3921, "label_cluster": -16},  # Messina
        {"latitudine": 45.8084, "longitudine": 9.1294, "label_cluster": -17},  # Como
        {"latitudine": 40.5550, "longitudine": 9.2750, "label_cluster": -18},  # Alghero
        {"latitudine": 41.1217, "longitudine": 16.8691, "label_cluster": -19},  # Bari
        {"latitudine": 41.0841, "longitudine": 14.7674, "label_cluster": -20},  # Campobasso
        {"latitudine": 44.1194, "longitudine": 9.1827, "label_cluster": -21},  # La Spezia
        {"latitudine": 43.9530, "longitudine": 12.9092, "label_cluster": -22},  # Perugia
        {"latitudine": 41.1227, "longitudine": 16.8663, "label_cluster": -23},  # Foggia
        {"latitudine": 45.3289, "longitudine": 10.6891, "label_cluster": -24},  # Mantua
        {"latitudine": 45.9254, "longitudine": 11.3025, "label_cluster": -25},  # Verona
        {"latitudine": 40.6495, "longitudine": 14.2642, "label_cluster": -26},  # Salerno
        {"latitudine": 43.0963, "longitudine": 12.6167, "label_cluster": -27},  # Ancona
        {"latitudine": 42.0195, "longitudine": 13.2300, "label_cluster": -28},  # Teramo
    ]
    env.update_environment(initial_data, step_title="Step 0: Raw Data Points")

    # Step 1: Initial Clustering - Assign some points to clusters (0, 1, 2)
    step_1_data = [
        # Northern Italy - Cluster 0
        {"latitudine": 45.4642, "longitudine": 9.1900, "label_cluster": 0},  # Milan
        {"latitudine": 44.4949, "longitudine": 11.3426, "label_cluster": 0},  # Bologna
        {"latitudine": 45.0703, "longitudine": 7.6869, "label_cluster": 0},  # Turin
        {"latitudine": 46.6034, "longitudine": 9.0149, "label_cluster": 0},  # Sondrio
        {"latitudine": 45.8084, "longitudine": 9.1294, "label_cluster": 0},  # Como
        {"latitudine": 45.9254, "longitudine": 11.3025, "label_cluster": 0},  # Verona
        {"latitudine": 43.7167, "longitudine": 10.4000, "label_cluster": 0},  # Pisa

        # Central Italy - Cluster 1
        {"latitudine": 43.7696, "longitudine": 11.2558, "label_cluster": 1},  # Florence
        {"latitudine": 42.3601, "longitudine": 13.4315, "label_cluster": 1},  # Pescara
        {"latitudine": 42.0560, "longitudine": 11.8818, "label_cluster": 1},  # Viterbo
        {"latitudine": 43.9530, "longitudine": 12.9092, "label_cluster": 1},  # Perugia
        {"latitudine": 42.0195, "longitudine": 13.2300, "label_cluster": 1},  # Teramo
        {"latitudine": 43.0963, "longitudine": 12.6167, "label_cluster": 1},  # Ancona
        {"latitudine": 40.6495, "longitudine": 14.2642, "label_cluster": 1},  # Salerno
        {"latitudine": 41.9028, "longitudine": 12.4964, "label_cluster": 1},  # Rome

        # Southern Italy - Cluster 2
        {"latitudine": 38.1157, "longitudine": 13.3615, "label_cluster": 2},  # Palermo
        {"latitudine": 37.7749, "longitudine": 15.0858, "label_cluster": 2},  # Catania
        {"latitudine": 40.8522, "longitudine": 14.2681, "label_cluster": 2},  # Naples
        {"latitudine": 38.0472, "longitudine": 15.3921, "label_cluster": 2},  # Messina
        {"latitudine": 40.8522, "longitudine": 14.2681, "label_cluster": 2},  # Naples
        {"latitudine": 41.1227, "longitudine": 16.8663, "label_cluster": 2},  # Foggia
        {"latitudine": 41.1217, "longitudine": 16.8691, "label_cluster": 2},  # Bari
        {"latitudine": 40.5550, "longitudine": 9.2750, "label_cluster": 2},  # Alghero
    ]
    env.update_environment(step_1_data, step_title="Step 1: Initial Clustering")

    # Step 2: Updated Clustering - Reassign some points to new clusters
    step_2_data = [
        # Northern Italy - Cluster 0
        {"latitudine": 45.4642, "longitudine": 9.1900, "label_cluster": 0},  # Milan
        {"latitudine": 44.4949, "longitudine": 11.3426, "label_cluster": 0},  # Bologna
        {"latitudine": 45.0703, "longitudine": 7.6869, "label_cluster": 0},  # Turin
        {"latitudine": 46.6034, "longitudine": 9.0149, "label_cluster": 0},  # Sondrio
        {"latitudine": 45.8084, "longitudine": 9.1294, "label_cluster": 0},  # Como
        {"latitudine": 45.9254, "longitudine": 11.3025, "label_cluster": 0},  # Verona
        {"latitudine": 43.7167, "longitudine": 10.4000, "label_cluster": 0},  # Pisa

        # Central Italy - Cluster 1
        {"latitudine": 43.7696, "longitudine": 11.2558, "label_cluster": 1},  # Florence
        {"latitudine": 42.3601, "longitudine": 13.4315, "label_cluster": 1},  # Pescara
        {"latitudine": 42.0560, "longitudine": 11.8818, "label_cluster": 1},  # Viterbo
        {"latitudine": 43.9530, "longitudine": 12.9092, "label_cluster": 1},  # Perugia
        {"latitudine": 42.0195, "longitudine": 13.2300, "label_cluster": 1},  # Teramo
        {"latitudine": 43.0963, "longitudine": 12.6167, "label_cluster": 1},  # Ancona
        {"latitudine": 40.6495, "longitudine": 14.2642, "label_cluster": 1},  # Salerno
        {"latitudine": 41.9028, "longitudine": 12.4964, "label_cluster": 1},  # Rome

        # Southern Italy - Cluster 2
        {"latitudine": 38.1157, "longitudine": 13.3615, "label_cluster": 2},  # Palermo
        {"latitudine": 37.7749, "longitudine": 15.0858, "label_cluster": 2},  # Catania
        {"latitudine": 40.8522, "longitudine": 14.2681, "label_cluster": 2},  # Naples
        {"latitudine": 38.0472, "longitudine": 15.3921, "label_cluster": 2},  # Messina
        {"latitudine": 41.1227, "longitudine": 16.8663, "label_cluster": 2},  # Foggia
        {"latitudine": 41.1217, "longitudine": 16.8691, "label_cluster": 2},  # Bari
        {"latitudine": 40.5550, "longitudine": 9.2750, "label_cluster": 2},  # Alghero
    ]
    env.update_environment(step_2_data, step_title="Step 2: Updated Clustering")

if __name__ == "__main__":
    main()

