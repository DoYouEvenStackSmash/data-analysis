#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <ctime>
#include <cstdlib>

#include <fstream>
#include <vector>
#include <sstream>
#include <iomanip>

#include "kmeans.h"

int main() {
    // Define the number of matrices and their dimensions
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
  // setup for k means clustering
  vector<int> data_refs;
  for (int i = 0; i < numMatrices; ++i)
    data_refs.push_back(i);

  vector<Eigen::MatrixXf> centroids;
  int starting_index = 2;
  int k = 12;
  kmeanspp_refs(&matrices[0], data_refs, starting_index, k, centroids);
  vector<vector<int>> ref_clusters;
  for (int i = 0; i < 100; ++i) {
    vector<Eigen::MatrixXf> new_ctr;
    vector<vector<int>> new_cls;
    kmeans_refs(&matrices[0], data_refs, centroids, true, new_ctr, new_cls);
    ref_clusters = new_cls;
    // ref_clusters = new_cls;
    if (new_ctr.size() < centroids.size()) {
      centroids = new_ctr;
      break;
    }
    centroids = new_ctr;
  }

  std::vector<std::vector<std::vector<double>>> data(ref_clusters.size());
  std::vector<std::vector<double>> data_centroids(ref_clusters.size());
  // for (int i = 0; i < ref_clusters.size(); )
  for (int i = 0; i < ref_clusters.size(); ++i) {
    int cidx = i;
    const Eigen::Map<const Eigen::VectorXf> flattened_centroid(centroids[cidx].data(), centroids[cidx].size());
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

