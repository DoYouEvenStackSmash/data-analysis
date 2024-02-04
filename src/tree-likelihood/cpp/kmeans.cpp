#include "kmeans.h"
#include <chrono>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
using Eigen::MatrixXd;
// This function performs the K-means clustering algorithm to assign data points
// to centroids. It calculates the distance between each data point and each
// centroid, assigning data points to the nearest centroid.
int kmeans_refs(MatrixXd *data_store, vector<int> &data_refs,
                 vector<MatrixXd> &centroids, bool FIRST_FLAG,
                 vector<MatrixXd> &new_centroids,
                 vector<vector<int>> &new_ref_clusters) {
  // Create a vector to store clusters of data point indices for each centroid.
  vector<vector<int>> ref_clusters;//(centroids.size());
  int counters[10];
  for (int i = 0; i < 10; ++i) {
    counters[i] = 0;
  }
  double min_dist;
  int nearest_neighbor;
  double pdist;

  // Loop through each data point and assign it to the nearest centroid.
  for (int didx = 0; didx < data_refs.size(); ++didx) {

    // Initialize variables to track the minimum distance and the nearest
    // centroid.
    min_dist = std::numeric_limits<double>::infinity();
    nearest_neighbor = -1;

    // Loop through each centroid and calculate the distance between the data
    // point and the centroid.
    for (int cidx = 0; cidx < centroids.size(); ++cidx) {
      // If it's the first iteration and FIRST_FLAG is true, skip comparing with
      // itself.
      if (FIRST_FLAG && data_store[data_refs[didx]] == centroids[cidx])
        continue;

      // Calculate the Euclidean distance between the data point and the
      // centroid.
      pdist = (data_store[data_refs[didx]] - centroids[cidx]).norm();

      // Update the nearest centroid if this centroid is closer.
      if (pdist < min_dist) {
        min_dist = pdist;
        nearest_neighbor = cidx;
      }
    }

    // If a nearest neighbor was found, assign the data point to the
    // corresponding cluster.
    if (nearest_neighbor != -1) {
      // if (!new_ref_clusters[nearest_neighbor].size())
      new_ref_clusters[nearest_neighbor][counters[nearest_neighbor]] = data_refs[didx];
      counters[nearest_neighbor]++;
      if (new_ref_clusters[nearest_neighbor].size() <= counters[nearest_neighbor])
        new_ref_clusters[nearest_neighbor].resize(new_ref_clusters[nearest_neighbor].size() * 2);
    }
  }

  // postprocessing for return
  // new_ref_clusters = ref_clusters;
  int counter = 0;
  for (int j = 0; j < new_ref_clusters.size(); ++j) {
    
    if (counters[j] > 0) {
      new_ref_clusters[j].resize(counters[j]);
      // new_ref_clusters.push_back(a);
      // Calculate the average of data points in the cluster to get the new
      // centroid.
      MatrixXd mean_mat(data_store[0].rows(), data_store[0].cols());
      mean_mat.setZero();
      for (int i = 0; i < counters[j]; ++i)
        mean_mat = mean_mat + data_store[new_ref_clusters[j][i]];
      mean_mat /= counters[j];
      new_centroids.push_back(mean_mat);
      counter++;
    }else {
      vector<int> a;
      new_ref_clusters[j] = a;
    }
  }
  return counter;
}

// This function performs the K-means++ initialization for centroids.
// It selects the initial centroids in a way that improves clustering performance.
void kmeanspp_refs(MatrixXd *data_store,
                   vector<int> &data_refs,
                   int starting_idx,
                   int k,
                   vector<MatrixXd> &centroids
                   ) {
  // Create a map to keep track of data points that have not been chosen as centroids.
  map<int, double> not_chosen;

  // Initialize the map with all data points except the starting point.
  for (int i = 0; i < data_refs.size(); ++i) {
    if (i == starting_idx)
      continue;
    not_chosen[i] = 0.0;
  }

  // Add the starting point as the first centroid.
  centroids.push_back(data_store[data_refs[starting_idx]]);

  double min_dist;
  double p_dist;
  double total_sum;
  int index;

  // Select k - 1 more centroids.
  for (int i = 0; i < k - 1; ++i) {
    total_sum = 0.0;
    index = -1;

    // Iterate through data points that have not been chosen as centroids.
    for (auto &didx : not_chosen) {
      min_dist = numeric_limits<double>::infinity();

      // Calculate the minimum distance to existing centroids for each data point.
      for (int cidx = 0; i < centroids.size(); ++i) {
        p_dist = (data_store[data_refs[didx.first]] - centroids[cidx]).norm();
        min_dist = min_dist < p_dist ? min_dist : p_dist;
      }

      // Assign the squared minimum distance to the data point in the map.
      not_chosen[didx.first] = pow(min_dist, 2);
      total_sum += not_chosen[didx.first];
    }

    // Choose the next centroid using weighted sampling.
    weighted_sample(not_chosen, total_sum, &index);

    // Add the selected data point as a centroid.
    centroids.push_back(data_store[data_refs[index]]);

    // Remove the chosen data point from the list of candidates.
    not_chosen.erase(index);

    // Break the loop if all data points have been chosen as centroids.
    if (not_chosen.empty())
      break;
  }
}


// This function performs weighted sampling to select an index based on provided weights.
// It generates a random value and selects an index proportionally to its weight.
void weighted_sample(map<int, double> &index_weight_map, double &sum_weights, int *return_index) {
    // Initialize a random number generator with a random seed.
    std::random_device rd;
    std::mt19937 gen(rd());

    // Create a uniform distribution for generating random values in the range [0.0, 1.0).
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Generate a random sample value in the range [0.0, 1.0).
    double sample_val = dist(gen);

    // Iterate through the provided map of indices and their associated weights.
    for (auto &a : index_weight_map) {
        // Normalize the weight and subtract it from the sample value.
        sample_val -= a.second / sum_weights;

        // If the sample value becomes less than or equal to 0, choose this index.
        if (sample_val <= 0.0) {
            *return_index = a.first; // Set the return index to the selected index.
            break; // Exit the loop once an index is chosen.
        }
    }

    // If no index was chosen (should not happen if weights are valid), select the last index.
    if (*return_index < 0) {
        auto last = index_weight_map.rbegin(); // Get the last element of the map.
        *return_index = last->first; // Set the return index to the last index.
    }
}
