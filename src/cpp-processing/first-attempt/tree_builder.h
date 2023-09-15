#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <limits>
#include <map>
#include <ctime>
#include <queue>
#include "CTNode.h"

std::map<int, CTNode*> construct_tree(Eigen::MatrixXf *data_store, int n,int k=3, int C=25, int R=100);