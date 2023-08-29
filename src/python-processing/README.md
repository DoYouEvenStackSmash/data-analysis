# Hierarchical Clustering with kmedioids, kmeans, and kmeanscpp

This repository contains a Python program for performing hierarchical clustering using various clustering algorithms such as kmedioids, kmeans, and kmeanscpp. The program allows you to build, load, and search hierarchical clustering structures based on the specified algorithms and parameters.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Build Hierarchical Clustering](#build-hierarchical-clustering)
  - [Load Hierarchical Clustering](#load-hierarchical-clustering)
  - [Search Hierarchical Clustering](#search-hierarchical-clustering)
- [Contributing](#contributing)
- [License](#license)

## Overview

This program implements a hierarchical clustering algorithm that supports multiple clustering algorithms, including kmedioids, kmeans, and kmeanscpp. It provides functionalities to build hierarchical clustering structures, load existing structures, and search for clusters based on input data.

## Getting Started

### Prerequisites

- Python 3.x
- NumPy
- argparse (already included in Python standard library)

### Installation

1. Clone this repository:

   ```sh
   git clone https://github.com/DoYouEvenStackSmash/data-analysis.git
   cd data-analysis/src/python-processing
   ```

2. Install the required dependencies using `pip`:

   ```sh
   pip install -r requirements.txt
   ```

## Usage

The program supports three main operations: building hierarchical clustering, loading existing clustering, and searching for clusters. You can use the command-line interface to perform these operations.

### Build Hierarchical Clustering

To build hierarchical clustering with different clustering parameters, use the following command:

```sh
python clustering_driver.py build -i example_2_2.npy -k 3 -R 30 -C 45 -o output_tree.json
```

Replace `example_2_2.npy` with the path to your input data file. The `-k`, `-R`, and `-C` flags allow you to specify the number of clusters, number of iterations, and cutoff value respectively. The `-o` flag is optional and can be used to specify an output file to save the hierarchical clustering structure.

### Load Hierarchical Clustering

To load an existing hierarchical clustering structure, use the following command:

```sh
python clustering_driver.py load -t existing_tree.json
```

Replace `existing_tree.json` with the path to the generated JSON file containing the hierarchy. Included in the JSON file is a `resources` field which includes the necessary support files to build the tree.

### Search Hierarchical Clustering

To search for clusters in an existing hierarchical clustering structure, use the following command:

```sh
python clustering_driver.py search -t existing_tree.json -M exampleM_2_2.npy -G
```

Replace `existing_tree.json` with the path to the JSON file containing the hierarchy and `exampleM_2_2.npy` with the path to the large input data file. The `-G` flag is optional and generates a graph from the tree data.

## Contributing

Contributions are welcome! If you have any ideas or improvements, please feel free to open issues or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.