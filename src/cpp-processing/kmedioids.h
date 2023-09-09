#include <iostream>                      // Standard I/O operations
#include <vector>                        // Dynamic arrays (std::vector)
#include <eigen3/Eigen/Dense>            // Eigen library for matrix operations
#include <ctime>                         // Time-related functions
#include <algorithm>                    // Algorithms (e.g., std::min_element)

// Function to assign data points to clusters based on distances to medioids.
// - 'distances' is the pairwise distance matrix between data points.
// - 'medioid_indices' contains the indices of current medioids.
// - 'ref_clusters' will store the assigned data point indices for each medioid.
void assign_clusters(
    Eigen::MatrixXf &distances,              // Pairwise distance matrix
    std::vector<int> &medioid_indices,      // Indices of current medioids
    std::vector<std::vector<int>> &ref_clusters  // Assigned data point indices for each medioid
);

// Function to compute the pairwise distance matrix between data points.
// - 'data' is an array of data points represented as Eigen::MatrixXf.
// - 'n' is the number of data points.
// - 'distances' is an output parameter to store the computed pairwise distance matrix.
void computeDistanceMatrix(
    Eigen::MatrixXf *data,                 // Array of data points
    int n,                                 // Number of data points
    Eigen::MatrixXf *distances             // Output parameter for pairwise distance matrix
);

// Function to preprocess data for k-medioids clustering.
// - 'data' is an array of data points represented as Eigen::MatrixXf.
// - 'n' is the number of data points.
// - 'k' is the number of medioids to be selected.
// - 'medioidIndices' will store the indices of selected medioids.
// - 'return_distances' is an output parameter to store the computed pairwise distance matrix.
void preprocess(
    Eigen::MatrixXf *data,                 // Array of data points
    int n,                                 // Number of data points
    int k,                                 // Number of medioids
    std::vector<int> &medioidIndices,      // Output: Indices of selected medioids
    Eigen::MatrixXf *return_distances     // Output: Pairwise distance matrix
);

// Function to update medioids based on assigned clusters.
// - 'distances' is the pairwise distance matrix between data points.
// - 'medioid_indices' contains the indices of current medioids.
// - 'ref_clusters' stores the assigned data point indices for each medioid.
void update_medioids(
    Eigen::MatrixXf &distances,              // Pairwise distance matrix
    std::vector<int> &medioid_indices,      // Indices of current medioids
    std::vector<std::vector<int>> &ref_clusters  // Assigned data point indices for each medioid
);

// Function to calculate the total sum of distances between data points and medioids.
// - 'distances' is the pairwise distance matrix between data points.
// - 'medioid_indices' contains the indices of current medioids.
// - 'ref_clusters' stores the assigned data point indices for each medioid.
// - 'total_sum' is an output parameter to store the calculated total sum.
void calculate_sum(
    Eigen::MatrixXf &distances,              // Pairwise distance matrix
    std::vector<int> &medioid_indices,      // Indices of current medioids
    std::vector<std::vector<int>> &ref_clusters,  // Assigned data point indices for each medioid
    double *total_sum                        // Output: Calculated total sum
);
