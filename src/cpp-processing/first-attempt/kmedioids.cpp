// #include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <ctime>
#include <algorithm>
#include <limits>
#include "kmedioids.h"
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
    MatrixXd* &distance_matrix,                 // Pairwise distance matrix
    std::vector<int> &medioid_indices,         // Indices of medioid points
    std::vector<std::vector<int>> &clusters, // Data point assignments to medioid clusters
    double *total_sum                          // Pointer to store the calculated sum
) {
    double sum_over_all = 0.0;

    // Iterate over medioid clusters
    for (int i = 0; i < medioid_indices.size(); ++i) {
        // Iterate over data points assigned to each medioid cluster
        for (int j = 0; j < clusters[i].size(); ++j) {
            // Accumulate the distance from the medioid to each assigned data point
            sum_over_all += (*distance_matrix)(medioid_indices[i], clusters[i][j]);
        }
    }

    // Store the calculated sum in the 'total_sum' pointer
    *total_sum = sum_over_all;
}


// Assigns data points to their nearest medioid clusters based on pairwise distances.
// This function computes distances between each data point and all medioids and assigns
// each data point to the cluster with the nearest medioid.
std::vector<std::vector<int>> assign_clusters(  MatrixXd* &distance_matrix,
                                                std::vector<int> &curr_medioids,
                                                int &n_elems) {
    std::vector<std::vector<int>> clusters(curr_medioids.size(), std::vector<int>());

    for (int i = 0; i < n_elems; ++i) {
        double min_dist = std::numeric_limits<double>::infinity();
        int min_idx = -1;
        double pdist = 0.0;
        for (int j = 0; j < curr_medioids.size(); ++j) {
            pdist = (*distance_matrix)(i, curr_medioids[j]);
            if (pdist < min_dist) {
                min_dist = pdist;
                min_idx = j;
            }
        }
        clusters[min_idx].push_back(i);
    }
    return clusters;

}


// Updates medioids for each cluster based on the sum of distances between data points in the cluster and all other data points.
// This function calculates a new medioid for each cluster by selecting the data point with the minimum sum of distances to all other data points in the cluster.

void update_medioids(   MatrixXd* &distance_matrix,
                        std::vector<std::vector<int>> clusters,
                        std::vector<int>& curr_medioids) {
    for (int i = 0; i < clusters.size(); ++i) {
        std::vector<double> D(clusters[i].size(), 0.0);
        for (int j = 0; j < clusters[i].size(); ++j) {
            for (int k = 0; k < clusters[i].size(); ++k) {
                D[j] += (*distance_matrix)(clusters[i][j], clusters[i][k]);
            }
        }
        int min_idx = 0;
        for (int c = 0; c < D.size(); ++c)
            min_idx = D[min_idx] < D[c] ? min_idx : c;
        curr_medioids[i] = clusters[i][min_idx];
    }
}


// Computes the pairwise distance matrix between data points.
// This function calculates the Euclidean distance between each pair of data points and stores the distances in a symmetric matrix.
void ComputeDistanceMatrix( MatrixXd *data_store,
                            std::vector<int> &data_refs,
                            MatrixXd* &distance_matrix
                            ) {

  for (int i = 0; i < data_refs.size(); ++i) {
    for (int j = i + 1; j < data_refs.size(); ++j) {
      double distance = (data_store[data_refs[i]] - data_store[data_refs[j]]).norm();
      (*distance_matrix)(i, j) = distance;
      (*distance_matrix)(j, i) = distance;
    }
  }
  // distance_matrix = pairwise_distances;
}

// Custom comparison function to compare P objects by vSum
bool compareByVSum(P& left,P& right) {
    return left.vSum < right.vSum;
}

void preprocess(MatrixXd *data_store, std::vector<int> &data_refs, MatrixXd* &distance_matrix, std::vector<int> &medioid_indices, int k) {
    // int n = Mlen;
    ComputeDistanceMatrix(data_store, data_refs, distance_matrix);

    Eigen::VectorXd denoms = distance_matrix->rowwise().sum();
    std::vector<P> v_vals(data_refs.size());
    for (int j = 0; j < data_refs.size(); ++j) {
        v_vals[j].idx = j;
        for (int i = 0; i < data_refs.size(); ++i) {
            v_vals[j].vSum += (*distance_matrix)(j, i) / denoms[i];
        }
    }

    std::sort(v_vals.begin(), v_vals.end(), compareByVSum);

    for (int i = 0; i < k; ++i)
        medioid_indices.push_back(v_vals[i].idx);
}
