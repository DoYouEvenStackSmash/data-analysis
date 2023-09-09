#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <ctime>
#include <algorithm>

// Assigns data points to clusters based on distances to medioids.
// Modifies the ref_clusters vector to store the assigned clusters.
void assign_clusters(
    Eigen::MatrixXf &distances,
    std::vector<int> &medioid_indices,
    std::vector<std::vector<int>> &ref_clusters
);

// Computes the pairwise distance matrix between data points and stores it in 'distances'.
void computeDistanceMatrix(
    Eigen::MatrixXf *data,
    int n,
    Eigen::MatrixXf *distances
);

// Preprocesses the data according to the k-medioids algorithm.
// Returns a list of indices referring to medioids in 'medioidIndices'.
// Computes the distance matrix and stores it in 'return_distances'.
void preprocess(
    Eigen::MatrixXf *data,
    int n,
    int k,
    std::vector<int> &medioidIndices,
    Eigen::MatrixXf *return_distances
);

// Updates the medioids based on the assigned clusters and distances.
void update_medioids(
    Eigen::MatrixXf &distances,
    std::vector<int> &medioid_indices,
    std::vector<std::vector<int>> &ref_clusters
);

// Calculates the total sum of distances between data points and their respective medioids.
// The result is stored in 'total_sum'.
void calculate_sum(
    Eigen::MatrixXf &distances,
    std::vector<int> &medioid_indices,
    std::vector<std::vector<int>> &ref_clusters,
    double *total_sum
);
