// #include <iostream>                      // Standard I/O operations
#include <vector>                        // Dynamic arrays (std::vector)
#include <eigen3/Eigen/Dense>            // Eigen library for matrix operations
#include <ctime>                         // Time-related functions
#include <algorithm>                    // Algorithms (e.g., std::min_element)
using Eigen::MatrixXd;

// Function to assign the data points to the nearest medioid
// elements are accessible as the integer indexes in distance_matrix
// Returns a vector of vectors representing the clusters
// - distance_matrix: reference to a caller allocated CxC matrix with (n_elems,n_elems) populated
// - curr_medioids: vector of integers representing the current medioids
// - n_elems: maximum dimensions of the square distance_matrix
std::vector<std::vector<int>> AssignClusters(  MatrixXd* &distance_matrix,
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
// - data_store: array of data points
// - data_refs: vector containing references to elements of data_store
// - distance_matrix: caller allocated MatrixXd for storing pairwise distances
// - medioid_indices: output vector which will contain medioid indices
// - k: number of clusters
void preprocess(
    MatrixXd *data_store,               // array of data points
    std::vector<int> &data_refs,        // array of data references
    MatrixXd* &distance_matrix,         // caller allocated pairwise distance matrix
    std::vector<int> &medioid_indices,  // Output: indices of data_refs
    int k                               // number of clusters
);

// Function to update medioids based on assigned clusters.

void UpdateMedioids(   MatrixXd* &distance_matrix,
                        std::vector<std::vector<int>> clusters,
                        std::vector<int>& curr_medioids);

// Function to calculate the total sum of distances between data points and medioids.
// - 'distances' is the pairwise distance matrix between data points.
// - 'medioid_indices' contains the indices of current medioids.
// - 'clusters' stores the assigned data point indices for each medioid.
// - 'total_sum' is an output parameter to store the calculated total sum.
void CalculateSum(
    MatrixXd* &distance_matrix,              // Pairwise distance matrix
    std::vector<int> &medioid_indices,      // Indices of current medioids
    std::vector<std::vector<int>> &clusters,  // Assigned data point indices for each medioid
    double *total_sum                        // Output: Calculated total sum
);
