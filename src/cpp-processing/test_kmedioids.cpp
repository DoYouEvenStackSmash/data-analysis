#include "kmedioids.h"
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iomanip>
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
    // Replace this with your actual data
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
  // for (int i = 0; i < ref_clusters.size(); )
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


    // Specify the output file path
    std::string outputFilePath = "output.json";

    // Open the output file for writing
    std::ofstream outputFile(outputFilePath);

    if (!outputFile.is_open()) {
        std::cerr << "Failed to open output file." << std::endl;
        return 1; // Exit with an error code
    }

    // Write the opening brace for the JSON array
    outputFile << "{\"clusters\": [";

    // Serialize each pair and write to the file
    int precision = 4;
    for (int i = 0; i < data.size(); ++i) {
      outputFile << "{\"" << i <<  "\": [";
      std::ostringstream jsonPair;
      double x = data_centroids[i][0];
      double y = data_centroids[i][1];
      jsonPair << "{\"x\": " << std::fixed << std::setprecision(precision) << x << ", \"y\": " << std::fixed << std::setprecision(precision) << y << "}";
      outputFile << jsonPair.str() << ",";
      for (const std::vector<double>& pair : data[i]) {
          x = pair[0];
          y = pair[1];

          // Create a JSON object for each pair
          std::ostringstream njsonPair;
          njsonPair << "{\"x\": " << std::fixed << std::setprecision(precision) << x << ", \"y\": " << std::fixed << std::setprecision(precision) << y << "}";

          // Write the JSON object to the file
          outputFile << njsonPair.str();

          // Add a comma separator if it's not the last pair
          if (&pair != &data[i].back()) {
              outputFile << ",";
          }
      }
      outputFile << "]}";
      if (i + 1 < data.size())
        outputFile << ",";
    }

    // Write the closing brace for the JSON array
    outputFile << "]}";

    // Close the output file
    outputFile.close();

    std::cout << "Serialization complete. Data written to " << outputFilePath << std::endl;

    return 0;
}