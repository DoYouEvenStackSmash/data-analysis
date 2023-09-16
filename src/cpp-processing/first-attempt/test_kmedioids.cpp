#include "kmedioids.h"
// #include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "aux_functions.h"
using namespace std;
using Eigen::MatrixXd;
int main() {

    vector<MatrixXd> data_store;
    vector<int> data_refs;
    
    mock_data_loader(data_store, data_refs);
    int C = data_refs.size();
    MatrixXd *dm = new MatrixXd(data_refs.size(), data_refs.size());
    dm->setZero();
    // ComputeDistanceMatrix(data_store.data(), data_refs, dm);
    vector<int> medioid_indices;
    int k = 5;
    preprocess(data_store.data(),data_refs, dm, medioid_indices, k);


    vector<vector<int>> ref_clusters;
    double total_sum = 0.0;
    for (int i = 0; i < 100; ++i) {
      double new_sum = 0.0;
      std::vector<std::vector<int>> temp_ref_clusters;//(medioidIndices.size(),vector<int>());
      temp_ref_clusters = AssignClusters(dm, medioid_indices, C);
      UpdateMedioids(dm, temp_ref_clusters, medioid_indices);

      CalculateSum(dm, medioid_indices, temp_ref_clusters, &new_sum);

      ref_clusters = temp_ref_clusters;
      if (new_sum == total_sum) {
        break;
      }
      total_sum = new_sum;
    }
  
  delete dm;
  return 0;
  std::vector<std::vector<std::vector<double>>> data(ref_clusters.size());
  std::vector<std::vector<double>> data_centroids(ref_clusters.size());
  int total = 0;
  for (int i = 0; i < ref_clusters.size(); ++i) {
    total += ref_clusters[i].size();
    int cidx = data_refs[medioid_indices[i]];
    const Eigen::Map<const Eigen::VectorXd> flattened_centroid(data_store[cidx].data(), data_store[cidx].size());
    data_centroids[i].assign(flattened_centroid.data(), flattened_centroid.data() + flattened_centroid.size());
    for (int j = 0; j < ref_clusters[i].size(); ++j) {
      int idx = data_refs[ref_clusters[i][j]];
      const Eigen::Map<const Eigen::VectorXd> flattened_vector(data_store[idx].data(), data_store[idx].size());
      std::vector<double> vec;
      vec.assign(flattened_vector.data(), flattened_vector.data() + flattened_vector.size());
      data[i].push_back(vec);
    }
  }
  printf("%d", total);
  delete dm;
  serialize_clusters(data, data_centroids);

  return 0;
}