#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <limits>
#include <map>
#include <ctime>
#include <queue>
using namespace std;


struct CTNode {
  Eigen::MatrixXf* val = nullptr;
  int _id = -1;
  vector<int> children;
  vector<int>* data = nullptr;
};

void foo(CTNode*& node) {
  CTNode* n = new CTNode();
  int numRows = 1;
  int numCols = 2;
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  Eigen::MatrixXf matrix(numRows, numCols);

  for (int row = 0; row < numRows; ++row) {
    for (int col = 0; col < numCols; ++col) {
      matrix(row, col) = static_cast<double>(std::rand()) / RAND_MAX * 100.0f;
    }
  }
  n->val = new Eigen::MatrixXf(matrix);
  n->data = new vector<int>(10);
  node = n;
}

int main() {

  map<int, CTNode*> node_map;
  int node_count = 0;
  for (int i = 0; i < 10; ++i) {
    CTNode* newNode = nullptr; // Initialize a pointer before passing to foo
    foo(newNode);
    newNode->_id = node_count;
    node_map[node_count] = newNode; // Store the pointer in the map
    node_count++;
  }
  for (auto& a : node_map) {
    cout << a.first << "\nid:\t" << a.second->_id << "\n" << *(a.second->val) << endl;
  }
  for (auto& a : node_map) {
    delete a.second->val; // Delete the Eigen matrix
    delete a.second->data;
    delete a.second;      // Delete the CTNode
    
  }
}