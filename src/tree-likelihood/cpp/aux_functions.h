#include <iostream>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <eigen3/Eigen/Dense>

#include <eigen3/Eigen/StdVector>

using Eigen::MatrixXd;
void serialize_clusters(std::vector<std::vector<std::vector<double>>> &data, std::vector<std::vector<double>> &data_centroids);

void mock_data_loader(std::vector<MatrixXd> &data_store, std::vector<int> &data_refs);
