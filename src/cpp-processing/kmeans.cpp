#include <iostream>
#include <eigen3/Eigen/Dense>
#include <random>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>
#include "kmeans.h"

// This function performs the K-means clustering algorithm to assign data points to centroids.
// It calculates the distance between each data point and each centroid, assigning data points to the nearest centroid.
void kmeans_refs(Eigen::MatrixXf *data_store,
                 vector<int> &data_refs,
                 vector<Eigen::MatrixXf> &centroids,
                 bool FIRST_FLAG,
                 vector<Eigen::MatrixXf> &new_centroids,
                 vector<vector<int>> &new_ref_clusters
                ) {
  // Create a vector to store clusters of data point indices for each centroid.
  vector<vector<int>> ref_clusters(centroids.size());

  double min_dist;
  int nearest_neighbor;
  double pdist;

  // Loop through each data point and assign it to the nearest centroid.
  for (int didx = 0; didx < data_refs.size(); ++didx) {
    
    // Initialize variables to track the minimum distance and the nearest centroid.
    min_dist = std::numeric_limits<double>::infinity();
    nearest_neighbor = -1;

    // Loop through each centroid and calculate the distance between the data point and the centroid.
    for (int cidx = 0; cidx < centroids.size(); ++cidx) {
      // If it's the first iteration and FIRST_FLAG is true, skip comparing with itself.
      if (FIRST_FLAG && data_store[data_refs[didx]] == centroids[cidx])
        continue;
      
      // Calculate the Euclidean distance between the data point and the centroid.
      pdist = (data_store[data_refs[didx]] - centroids[cidx]).norm();

      // Update the nearest centroid if this centroid is closer.
      if (pdist < min_dist) {
        min_dist = pdist;
        nearest_neighbor = cidx;
      }
    }

    // If a nearest neighbor was found, assign the data point to the corresponding cluster.
    if (nearest_neighbor != -1) {
      ref_clusters[nearest_neighbor].push_back(data_refs[didx]);
    }
  }


  
  // postprocessing for return
  for (vector<int> a : ref_clusters) {
    if (a.size()) {
      new_ref_clusters.push_back(a);
      // Calculate the average of data points in the cluster to get the new centroid.
      Eigen::MatrixXf mean_mat(data_store[0].rows(),data_store[0].cols());
      mean_mat.setZero();
      for (int i = 0; i < a.size(); ++i)
        mean_mat = mean_mat + data_store[a[i]];
      mean_mat /= a.size();
      new_centroids.push_back(mean_mat);
    }
  }
}

