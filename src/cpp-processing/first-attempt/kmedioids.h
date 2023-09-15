// #include <iostream>                      // Standard I/O operations
#include <vector>                        // Dynamic arrays (std::vector)
#include <eigen3/Eigen/Dense>            // Eigen library for matrix operations
#include <ctime>                         // Time-related functions
#include <algorithm>                    // Algorithms (e.g., std::min_element)
using Eigen::MatrixXd;

std::vector<std::vector<int>> assign_clusters(  MatrixXd* &distance_matrix,
                                                std::vector<int> &curr_medioids,
                                                int &n_elems);

// function to compute pairwise distances between elements of data_store, referred to by 
// elements of data_refs.
// - data_store : Pointer to an array of data
// - data_refs: Reference to an array of datum indices
// - distance_matrix: Pointer to a caller allocated matrix to be populated by this function.
void ComputeDistanceMatrix( MatrixXd *data_store,
                            std::vector<int> &data_refs,
                            MatrixXd* &distance_matrix
                            );
// Function to preprocess data for k-medioids clustering.
// - 'data' is an array of data points represented as Eigen::MatrixXf.
// - 'medioidIndices' will store the indices of selected medioids.
// - 'return_distances' is an output parameter to store the computed pairwise distance matrix.
void preprocess(
    MatrixXd *data_store,                 // Array of data points
    std::vector<int> &data_refs,
    MatrixXd* &distance_matrix,                                // Number of medioids
    std::vector<int> &medioid_indices,
    int k
);

// Function to update medioids based on assigned clusters.

void update_medioids(   MatrixXd* &distance_matrix,
                        std::vector<std::vector<int>> clusters,
                        std::vector<int>& curr_medioids);
// Function to calculate the total sum of distances between data points and medioids.
// - 'distances' is the pairwise distance matrix between data points.
// - 'medioid_indices' contains the indices of current medioids.
// - 'ref_clusters' stores the assigned data point indices for each medioid.
// - 'total_sum' is an output parameter to store the calculated total sum.
void calculate_sum(
    MatrixXd* &distance_matrix,              // Pairwise distance matrix
    std::vector<int> &medioid_indices,      // Indices of current medioids
    std::vector<std::vector<int>> &clusters,  // Assigned data point indices for each medioid
    double *total_sum                        // Output: Calculated total sum
);
