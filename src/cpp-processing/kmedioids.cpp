#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <ctime>
#include <algorithm>

struct P {
  int idx;
  double vSum;
};

void calculate_sum(Eigen::MatrixXf &distances, std::vector<int> &medioid_indices, std::vector<std::vector<int>> &ref_clusters, double *total_sum) {
  double sum_over_all = 0.0;
  for (int i = 0; i < medioid_indices.size(); ++i) {
    for (int j = 0; j < ref_clusters[i].size(); ++j) {
      sum_over_all += distances(medioid_indices[i], ref_clusters[i][j]);
    }
  }
  *total_sum = sum_over_all;
}

void assign_clusters(Eigen::MatrixXf &distances, std::vector<int> &medioid_indices, std::vector<std::vector<int>> &ref_clusters) {
  int n = distances.rows();
  int m = medioid_indices.size();
  std::vector<double> D(m);
  int idx;
  for (int didx = 0; didx < n; ++didx) {
    for (int midx = 0; midx < m; ++midx) {
      D[midx] = distances(didx, medioid_indices[midx]);
    }
    idx = std::distance(D.begin(),std::min_element(D.begin(), D.end()));
    ref_clusters[idx].push_back(didx);
  }
}

void update_medioids(Eigen::MatrixXf &distances, std::vector<int> &medioid_indices, std::vector<std::vector<int>> &ref_clusters) {
  for (int i = 0; i < medioid_indices.size(); ++i) {
    std::vector<double> D(ref_clusters[i].size());
    
    for (int j = 0; j < ref_clusters[i].size(); ++j){
      int center_pt = ref_clusters[i][j];
      for (int k = 0; k < ref_clusters[i].size(); ++k) {
        D[j] += distances(center_pt, ref_clusters[i][k]);
      }
    }
    medioid_indices[i] = ref_clusters[i][std::distance(D.begin(),std::min_element(D.begin(), D.end()))];
  }
}

void computeDistanceMatrix(Eigen::MatrixXf *data, int n, Eigen::MatrixXf *return_distances) {
    
    Eigen::MatrixXf pairwiseDistances(n, n);

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double distance = (data[i] - data[j]).norm();
            pairwiseDistances(i, j) = distance;
            pairwiseDistances(j, i) = distance;
        }
    }

    *return_distances = pairwiseDistances;
}


void preprocess(Eigen::MatrixXf *data, int n, int k, std::vector<int> &medioidIndices, Eigen::MatrixXf *distances) {
    // int n = Mlen;

    // Compute pairwise distance matrix
    computeDistanceMatrix(data,n,distances);

    // Step 1-2: Calculate denominators efficiently
    Eigen::VectorXf denominators = (*distances).rowwise().sum();

    // Calculate v values
    Eigen::MatrixXf vValues = (*distances).array().colwise() / denominators.array();

    // Set diagonal values to 0
    vValues.diagonal().setZero();

    Eigen::VectorXf vSums = vValues.rowwise().sum();

    // Initialize data as pairs of index and v value
    std::vector<P> v_arr(n);
    for (int idx = 0; idx < n; ++idx) {
        v_arr[idx].idx = idx;
        v_arr[idx].vSum = vSums[idx];
        // data[vSums] = &vSums[idx];
    }

    // Sort data by v values
    std::sort(v_arr.begin(), v_arr.end(), [](const auto& left, const auto& right) {
        return (left.vSum) < (right.vSum);
    });


    // Get the indices of the k medioids
    // std::vector<int> medioidIndices;
    for (int i = 0; i < k; ++i) {
        medioidIndices.push_back(v_arr[i].idx);
    }
}
