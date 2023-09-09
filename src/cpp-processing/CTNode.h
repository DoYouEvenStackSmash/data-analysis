#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>

using namespace std;
struct CTNode {
  Eigen::MatrixXf* val = nullptr;
  int _id = -1;
  vector<int> children;
  vector<int>* data = nullptr;
};