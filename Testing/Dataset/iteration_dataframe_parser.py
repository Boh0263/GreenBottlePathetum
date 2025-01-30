import pandas as pd

def load_and_parse_iterations(csv_file_path):

    df = pd.read_csv(csv_file_path)

    if 'iteration' not in df.columns:
        print(f"Error: The column 'iteration' does not exist in the CSV file at {csv_file_path}.")
        return None

    iterations_data = {}

    iterations = df['iteration'].unique()

    for iteration in iterations:
        iteration_df = df[df['iteration'] == iteration].copy()
        iterations_data[iteration] = iteration_df

        print(f"Iteration {iteration} Data:\n", iteration_df.head(), "\n")

    return iterations_data