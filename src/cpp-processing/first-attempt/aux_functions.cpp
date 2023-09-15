#include <ctime>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <eigen3/Eigen/Dense>

#include <eigen3/Eigen/StdVector>
#include "aux_functions.h"
using Eigen::MatrixXd;
void serialize_clusters(std::vector<std::vector<std::vector<double>>> &data, std::vector<std::vector<double>> &data_centroids) {

    // Specify the output file path
    std::string outputFilePath = "output.json";

    // Open the output file for writing
    std::ofstream outputFile(outputFilePath);

    if (!outputFile.is_open()) {
        std::cerr << "Failed to open output file." << std::endl;
        // return 1; // Exit with an error code
        return;
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

}
  

void mock_data_loader(std::vector<MatrixXd> &data_store, std::vector<int> &data_refs) {
    std::srand(12345);
    double max_val = 100.0;
    int r = 1;
    int c = 2;
    int num_matrices = 100;

    for (int i = 0; i < num_matrices; ++i) {
        MatrixXd m(r,c);
        m.setZero();
        for (int row = 0; row < r; ++row) {
            for (int col = 0; col < c; ++col) {
                m(row,col) = static_cast<double>(std::rand()) / RAND_MAX * max_val;
            }
        }
        data_store.push_back(m);
        data_refs.push_back(i);
    }
}