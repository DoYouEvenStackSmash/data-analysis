#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <limits>
#include <map>
#include <ctime>
#include <queue>
#include "CTNode.h"
using Eigen::MatrixXd;
// Function to construct a hierarchical clustering over the elements of data_store
// 
std::map<int, CTNode*> construct_tree(MatrixXd *data_store, vector<int> &init_data_refs,int k=3, int C=25, int R=100);