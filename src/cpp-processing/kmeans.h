#include <iostream>
#include <eigen3/Eigen/Dense>
#include <random>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>
using namespace std;
void kmeans_refs(Eigen::MatrixXf *data_store,
                                vector<int> &data_refs,
                                vector<Eigen::MatrixXf> &centroids,
                                bool FIRST_FLAG,
                                vector<Eigen::MatrixXf> &new_centroids,
                                vector<vector<int>> &new_ref_clusters
                                );