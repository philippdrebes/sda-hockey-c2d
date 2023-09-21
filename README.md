# Sport Data Analytics for Swiss Ice Hockey

## Introduction

This repository contains the codebase for analyzing player performance in Swiss ice hockey using the
innovative [compute to data](https://docs.oceanprotocol.com/developers/compute-to-data) approach. By utilizing this
method, we aim to ensure data privacy and security while still enabling a
comprehensive data analysis by comparing individual player data to the pooled data of the entire league. This approach
allows for data analysis without needing direct access to the sensitive player data.

## Repository Structure

### File/Folder Descriptions

- **algos**:
    - `hockey.py`: Contains the algorithm for our hockey data analysis. This algorithm is published on the blockchain
      and runs in the compute to data environment.

- **c2d**:
    - `dispatcher.py`: This is a helper file containing functions crucial for the compute to data workflow. It has
      functionalities to:
        - Publish the data and the algorithm onto the blockchain.
        - Allow the algorithm to execute on the data.

- **notebooks**: Contains Jupyter Notebooks that were used during the development and exploratory phase of the project.
    - `data_per_period.ipynb`: Explores and analyzes the data on a period-by-period basis.
    - `dummy_data_creation.ipynb`: Notebooks used to generate the dummy data for testing.
    - `strenghts_weaknesses.ipynb`: Analyzes the strengths and weaknesses of players based on the given metrics.

- `example.env`: An example environment file showcasing the environment variables required to run the project.

- `requirements.txt`: Contains a list of necessary Python packages required to run the project.

- `Dockerfile`: In the compute to data space, every algorithm runs within a Docker container to ensure consistency and
  reproducibility. While there are default images provided for this purpose, there are instances where the dependencies
  and configurations required for a project might not be available in these default images. This is where our custom
  Docker image comes into play. The `Dockerfile` contains instructions to build our own Docker image, tailored
  specifically for our project.

- `main.py`: The central entry point of the project. This script is responsible for the orchestration of the primary
  compute-to-data tasks:
    - Publishing the algorithm and data onto the blockchain.
    - Initiating a compute job that executes the algorithm on the provided data.
    - Retrieving the results post-computation.

## Getting Started

1. Clone this repository to your local machine.
2. Install the required packages using: `pip install -r requirements.txt`.
3. Navigate to the main directory and run: `python main.py`.

