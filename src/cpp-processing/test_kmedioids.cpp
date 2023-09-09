#include "kmedioids.h"
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "aux_functions.h"
using namespace std;
int main() {
    // Example usage:
    int k = 12;
    const int numMatrices = 1000;
    const int numRows = 1;
    const int numCols = 2;

    // Create a vector to store the Eigen matrices
    // std::vector<Eigen::MatrixXf> matrices;
    Eigen::MatrixXf matrices[numMatrices];

    // Seed the random number generator
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Generate and populate the matrices
    for (int i = 0; i < numMatrices; ++i) {
        Eigen::MatrixXf matrix(numRows, numCols);

        // Populate the matrix with random values between 0 and 100
        for (int row = 0; row < numRows; ++row) {
            for (int col = 0; col < numCols; ++col) {
                matrix(row, col) = static_cast<double>(std::rand()) / RAND_MAX * 100.0f;
            }
        }
        matrices[i].resize(numRows, numCols);
        matrices[i] = matrix;
        // matrices.push_back(matrix);
    }
    
    std::vector<int> medioidIndices;
    Eigen::MatrixXf distances;
    preprocess(&matrices[0], numMatrices ,k, medioidIndices, &distances);
    vector<vector<int>> ref_clusters;
    double total_sum = 0.0;
    for (int i = 0; i < 100; ++i) {
      double new_sum = 0.0;
      vector<vector<int>> temp_ref_clusters(medioidIndices.size());
      assign_clusters(distances, medioidIndices, temp_ref_clusters);
      update_medioids(distances, medioidIndices, temp_ref_clusters);
      calculate_sum(distances, medioidIndices, temp_ref_clusters, &new_sum);
      ref_clusters = temp_ref_clusters;
      if (new_sum == total_sum) {
        break;
      }
      total_sum = new_sum;
    }
  
    
  std::vector<std::vector<std::vector<double>>> data(ref_clusters.size());
  std::vector<std::vector<double>> data_centroids(ref_clusters.size());
  
  for (int i = 0; i < ref_clusters.size(); ++i) {
    int cidx = medioidIndices[i];
    const Eigen::Map<const Eigen::VectorXf> flattened_centroid(matrices[cidx].data(), matrices[cidx].size());
    data_centroids[i].assign(flattened_centroid.data(), flattened_centroid.data() + flattened_centroid.size());
    for (int j = 0; j < ref_clusters[i].size(); ++j) {
      int idx = ref_clusters[i][j];
      const Eigen::Map<const Eigen::VectorXf> flattened_vector(matrices[idx].data(), matrices[idx].size());
      std::vector<double> vec;
      vec.assign(flattened_vector.data(), flattened_vector.data() + flattened_vector.size());
      data[i].push_back(vec);
    }
  }

  serialize_clusters(data, data_centroids);

  return 0;
}