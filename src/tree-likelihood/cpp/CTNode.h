#ifndef CTNODE_H
#define CTNODE_H
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
using Eigen::MatrixXd;
using namespace std;

struct CTNode {
  Eigen::MatrixXd* val = nullptr;
  int _id = -1;
  vector<int> children;
  vector<int>* data = nullptr;
};

inline bool isLeafNode(CTNode* a) {
  return (a->data == nullptr);
}

inline bool isInteriorNode(CTNode* a) {
  return !isLeafNode(a);
}

#endif