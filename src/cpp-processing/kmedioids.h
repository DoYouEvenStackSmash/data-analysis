#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <ctime>
#include <algorithm>

void assign_clusters( Eigen::MatrixXf &distances, 
                      std::vector<int> &medioid_indices, 
                      std::vector<std::vector<int>> &ref_clusters
                      );
void computeDistanceMatrix( Eigen::MatrixXf *data, 
                            int n, 
                            Eigen::MatrixXf *distances);

void preprocess(  Eigen::MatrixXf *data, 
                  int n, 
                  int k, 
                  std::vector<int> &medioidIndices, 
                  Eigen::MatrixXf *return_distances);

void update_medioids( Eigen::MatrixXf &distances, 
                      std::vector<int> &medioid_indices, 
                      std::vector<std::vector<int>> &ref_clusters);

void calculate_sum( Eigen::MatrixXf &distances, 
                    std::vector<int> &medioid_indices, 
                    std::vector<std::vector<int>> &ref_clusters, 
                    double *total_sum);