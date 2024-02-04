// #include <iostream>
#include <eigen3/Eigen/Dense>
#include <random>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>
#include <map>

using namespace std;
using Eigen::MatrixXd;
// Function: kmeans_refs
// Description: Performs the K-means clustering algorithm to assign data points to centroids.
//              It calculates the distance between each data point and each centroid,
//              assigning data points to the nearest centroid.
// Parameters:
//   - data_store: Pointer to an array of data points (matrices).
//   - data_refs: Reference to a vector of data point indices.
//   - centroids: Reference to a vector of current centroids.
//   - FIRST_FLAG: A boolean flag indicating whether it's the first iteration.
//   - new_centroids: Reference to a vector to store the newly computed centroids.
//   - new_ref_clusters: Reference to a vector of vectors to store data point assignments to clusters.
int kmeans_refs(MatrixXd *data_store,
                 vector<int> &data_refs,
                 vector<MatrixXd> &centroids,
                 bool FIRST_FLAG,
                 vector<MatrixXd> &new_centroids,
                 vector<vector<int>> &new_ref_clusters);

// Function: kmeanspp_refs
// Description: Performs K-means++ initialization for centroids.
//              It selects initial centroids in a way that improves clustering performance.
// Parameters:
//   - data_store: Pointer to an array of data points (matrices).
//   - data_refs: Reference to a vector of data point indices.
//   - starting_idx: Index of the first centroid.
//   - k: The number of centroids to select.
//   - centroids: Reference to a vector to store the selected centroids.
void kmeanspp_refs(MatrixXd *data_store,
                   vector<int> &data_refs,
                   int starting_idx,
                   int k,
                   vector<MatrixXd> &centroids);

// Function: weighted_sample
// Description: Performs weighted sampling to select an index based on provided weights.
//              It generates a random value and selects an index proportionally to its weight.
// Parameters:
//   - index_weight_map: Reference to a map containing indices as keys and associated weights as values.
//   - total_sum: Reference to the total sum of weights.
//   - return_index: Pointer to an integer to store the selected index.
void weighted_sample(map<int, double> &index_weight_map, double &total_sum, int *return_index);
