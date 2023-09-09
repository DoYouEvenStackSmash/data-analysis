#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <ctime>
#include <algorithm>

struct P {
  int idx = 0;
  double vSum = 0.0;
};

// Calculates the total sum of distances between data points and their respective medioids.
// This function iterates over medioid clusters and their assigned data points.
// - 'distances' is the pairwise distance matrix between data points and medioids.
// - 'medioid_indices' is a vector containing indices of medioid points.
// - 'ref_clusters' is a vector of vectors containing indices of data points assigned to each medioid cluster.
// - 'total_sum' is a pointer to a double where the calculated sum will be stored.
void calculate_sum(
    Eigen::MatrixXf &distances,                 // Pairwise distance matrix
    std::vector<int> &medioid_indices,         // Indices of medioid points
    std::vector<std::vector<int>> &ref_clusters, // Data point assignments to medioid clusters
    double *total_sum                          // Pointer to store the calculated sum
) {
    double sum_over_all = 0.0;

    // Iterate over medioid clusters
    for (int i = 0; i < medioid_indices.size(); ++i) {
        // Iterate over data points assigned to each medioid cluster
        for (int j = 0; j < ref_clusters[i].size(); ++j) {
            // Accumulate the distance from the medioid to each assigned data point
            sum_over_all += distances(medioid_indices[i], ref_clusters[i][j]);
        }
    }

    // Store the calculated sum in the 'total_sum' pointer
    *total_sum = sum_over_all;
}


// Assigns data points to their nearest medioid clusters based on pairwise distances.
// This function computes distances between each data point and all medioids and assigns
// each data point to the cluster with the nearest medioid.
// - 'distances' is the pairwise distance matrix between data points and medioids.
// - 'medioid_indices' is a vector containing indices of medioid points.
// - 'ref_clusters' is a vector of vectors where data point indices assigned to each cluster will be stored.
void assign_clusters(
    Eigen::MatrixXf &distances,                 // Pairwise distance matrix
    std::vector<int> &medioid_indices,         // Indices of medioid points
    std::vector<std::vector<int>> &ref_clusters // Data point assignments to medioid clusters
) {
    int n = distances.rows(); // Number of data points
    int m = ref_clusters.size(); // Number of medioids
    std::vector<double> D(m,0.0); // Distances from a data point to all medioids
    int idx  = 0; // Index of the nearest medioid

    // Iterate over all data points
    for (int didx = 0; didx < n; ++didx) {
        // Calculate distances from the current data point to all medioids
        for (int midx = 0; midx < m; ++midx) {
            D[midx] = distances(didx, medioid_indices[midx]);
        }

        // Find the index of the nearest medioid (cluster)
      // if (D.size() == 1) {
        // std::cout <<"error"<< std::endl;
        // ref_clusters[0].push_back(didx);
      // }else {
        auto curr_posn = std::min_element(D.begin(), D.end());
        idx = std::distance(D.begin(), curr_posn);
        ref_clusters[idx].push_back(didx);
      // }

        // Assign the current data point to the cluster with the nearest medioid
        
    }
}


// Updates medioids for each cluster based on the sum of distances between data points in the cluster and all other data points.
// This function calculates a new medioid for each cluster by selecting the data point with the minimum sum of distances to all other data points in the cluster.
// - 'distances' is the pairwise distance matrix between data points.
// - 'medioid_indices' is a vector containing the current indices of medioid points.
// - 'ref_clusters' is a vector of vectors representing the data point assignments to each cluster.
void update_medioids(
    Eigen::MatrixXf &distances,                 // Pairwise distance matrix
    std::vector<int> &medioid_indices,         // Indices of current medioid points
    std::vector<std::vector<int>> &ref_clusters // Data point assignments to medioid clusters
) {
    // Iterate over each cluster
    for (int i = 0; i < medioid_indices.size(); ++i) {
        std::vector<double> D(ref_clusters[i].size()); // Distances of data points in the cluster to others

        // Calculate the sum of distances for each data point in the cluster
        for (int j = 0; j < ref_clusters[i].size(); ++j) {
            int center_pt = ref_clusters[i][j];
            for (int k = 0; k < ref_clusters[i].size(); ++k) {
                D[j] += distances(center_pt, ref_clusters[i][k]);
            }
        }

        // Update the medioid index for the current cluster by selecting the data point
        // with the minimum sum of distances as the new medioid.
        medioid_indices[i] = ref_clusters[i][std::distance(D.begin(), std::min_element(D.begin(), D.end()))];
    }
}


// Computes the pairwise distance matrix between data points.
// This function calculates the Euclidean distance between each pair of data points and stores the distances in a symmetric matrix.
// - 'data' is an array of data points represented as Eigen::MatrixXf.
// - 'n' is the number of data points.
// - 'return_distances' is an output parameter that will contain the computed pairwise distance matrix.
void computeDistanceMatrix(
    Eigen::MatrixXf *data,           // Array of data points
    int n,                           // Number of data points
    Eigen::MatrixXf &return_distances // Output parameter for pairwise distance matrix
) {
    // Initialize a matrix to store pairwise distances
    Eigen::MatrixXf pairwiseDistances(n, n);

    // Iterate over each pair of data points
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            // Calculate the Euclidean distance between data[i] and data[j]
            double distance = (data[i] - data[j]).norm();

            // Store the distance in the pairwise distance matrix
            pairwiseDistances(i, j) = distance;
            pairwiseDistances(j, i) = distance;
        }
    }

    // Set the 'return_distances' matrix to the computed pairwise distance matrix
    return_distances = pairwiseDistances;
}

// Custom comparison function to compare P objects by vSum
bool compareByVSum(P& left,P& right) {
    return left.vSum < right.vSum;
}

void preprocess(Eigen::MatrixXf *data, int n, int k, std::vector<int> &medioidIndices, Eigen::MatrixXf &distances) {
    // int n = Mlen;

    // Compute pairwise distance matrix
    computeDistanceMatrix(data,n,distances);

    // Step 1-2: Calculate denominators efficiently
    Eigen::VectorXf denominators = distances.rowwise().sum();

    // Calculate v values
    Eigen::MatrixXf vValues = distances.array().colwise() / denominators.array();

    // Set diagonal values to 0
    vValues.diagonal().setZero();

    Eigen::VectorXf vSums = vValues.rowwise().sum();

    // Initialize data as pairs of index and v value
    std::vector<P> v_arr(n,{0,0.0});
    for (int idx = 0; idx < n; ++idx) {
        v_arr[idx].idx = idx;
        v_arr[idx].vSum = vSums[idx];
        // data[vSums] = &vSums[idx];
    }

    // Sort data by v values
    std::sort(v_arr.begin(), v_arr.end(), compareByVSum);


    // Get the indices of the k medioids
    // std::vector<int> medioidIndices;
    for (int i = 0; i < k; ++i) {
        medioidIndices.push_back(v_arr[i].idx);
    }
}
