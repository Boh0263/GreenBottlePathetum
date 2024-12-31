# GreenBottlePathetum

GreenBottlePathetum is an AI-driven clustering system developed for the optimization of delivery services in a green logistics context. This project aims to support AquaPure's GreenBottle platform by utilizing clustering algorithms to efficiently assign delivery orders to couriers, optimize routes, and minimize operational costs.

## Features

- **Order Clustering**: Groups delivery orders into optimal clusters for assignment to couriers.
- **Route Optimization**: Suggests the best delivery routes for each courier.
- **Courier Load Balancing**: Ensures a balanced workload among couriers.
- **Scalability**: Efficiently handles various sizes of datasets and supports different geographical areas.

## Background and Goals

The platform GreenBottle, owned by AquaPure, specializes in delivering beverages in glass bottles using electric vehicles. This project leverages artificial intelligence to:
- Group orders optimally among available couriers.
- Determine the best routes for couriers to follow.
- Calculate the ideal number of couriers required to fulfill incoming orders, even if this exceeds the available workforce.

This clustering approach addresses the problem of categorizing orders and mapping optimal delivery routes, considering both efficiency and environmental sustainability.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Boh0263/GreenBottlePathetum.git
   ```

2. Navigate to the project directory:

   ```bash
   cd GreenBottlePathetum
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Testing

To validate the system functionality, follow these steps:

1. **Clone the repository (if not done already)**:

   ```bash
   git clone https://github.com/Boh0263/GreenBottlePathetum.git
   ```

2. **Navigate to the testing directory**:

   ```bash
   cd GreenBottlePathetum/Testing
   ```

3. **Install dependencies**:

   ```bash
   pip install -r ../requirements.txt
   ```
4. **Set the RANDOM.ORG API Key (Optional)**:
   The project uses the RANDOM.ORG JSON-RPC API to randomize datasets. 
   If the API key is not set, the generator uses Python's 'random' module as a fallback method.

   - **Access the Dataset Directory**:
          ```bash
          cd GreenBottlePathetum/Testing/Dataset
          ```
   - **Edit the `random_config.json` file**:
      Open the `random_config.json` file in a text editor (e.g., nano, vim, or a graphical text editor like VSCode).
      Find the `api_key` key inside the file. It should appear like the following example:
       ```nano
      {   
          "api_key": "YOUR_RANDOM_ORG_API_KEY_HERE"
      }
      ```
   - **Get an API Key**:
       Go to RANDOM.ORG and sign up to get an API key (if you don't have one already).
       Replace "YOUR_RANDOM_ORG_API_KEY" with your actual key.

   - **Save and Close the File**:
      Once you've added the API key, save the changes to the `random_config.json` file and close the editor.
5. **Execute the test script**:

   ```bash
   python main.py
   ```

This will run the clustering system on a sample dataset and demonstrate its performance.

## How It Works

### Data Preprocessing
The system uses a dataset containing:
- Order information (recipient details, delivery address, and order content).
- Courier data (number of couriers and capacity).

Preprocessing steps include:
1. Mapping delivery addresses to a Cartesian plane (e.g., converting addresses into coordinates for better route calculation).
2. Filtering out irrelevant data and normalizing geographic coordinates.
3. Structuring orders into a usable format for clustering algorithms.

### Clustering Algorithms
The project evaluates different clustering algorithms:
- **K-Means**: Chosen for its computational efficiency and suitability for fixed cluster numbers.
- **Hierarchical Clustering**: Offers better visualization but is computationally expensive for large datasets.
- **DBSCAN**: Useful for detecting outliers but less efficient for the primary requirements.

**Conclusion**: K-Means is identified as the best fit for the project due to its scalability and speed.

### Integration
Future improvements will focus on integrating the clustering system with AquaPure's existing logistics platform. Features under development include:
- An admin panel for selecting specific orders to cluster.
- Adapter modules to interface with AquaPure's backend.

## Future Enhancements
- Enable admins to manually choose which orders to process through the clustering system.
- Extend clustering functionalities to incorporate real-time traffic data for dynamic route optimization.

## Contributions
Contributions to the project are welcome. Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the LICENSE file for more information.
