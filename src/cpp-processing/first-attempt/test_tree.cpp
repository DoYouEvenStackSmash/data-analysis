#include "kmedioids.h"
#include "kmeans.h"
#include "tree_builder.h"
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <chrono>
// #include "CTNode.h"

int main() {
    const int numMatrices = 20;
    const int numRows = 1;
    const int numCols = 2;

    // Create a vector to store the Eigen matrices
    // std::vector<Eigen::MatrixXf> matrices;
    // Allocate memory for an array of Eigen::MatrixXf
    Eigen::MatrixXf* matrices = new Eigen::MatrixXf[numMatrices];

    // Seed the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 100.0);

    // Initialize each matrix with random values
    for (int i = 0; i < numMatrices; ++i) {
        matrices[i].resize(numRows, numCols); // Resize the matrix if needed
        for (int j = 0; j < numRows; ++j) {
            for (int k = 0; k < numCols; ++k) {
                matrices[i](j, k) = dist(gen); // Initialize with random value
            }
        }
    }

    // auto start = std::chrono::steady_clock::now();
    map<int,CTNode*> node_map = construct_tree(matrices, numMatrices, 3, 6, 30);
    // cout <<"done";
    // auto end = std::chrono::steady_clock::now();
    // auto time_diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    // std::cout << "time:\t" << time_diff / 1e9f << "\n";//<< "total:\t" << total << "\n";
    for (auto& a : node_map) {
      cout << "id:\t" << a.first << endl;
      for (auto &b : a.second->children) {
        cout << "\t" << b << endl;
      }
      if (a.first != -1) {
        cout << a.first << "\nid:\t" << a.second->_id << "\n" << *(a.second->val) << endl;
        if (a.second->data) {
          cout << "[";
          for (auto &b : (*a.second->data)) {
            cout << b << ",";
          }
          cout << "];\n";
        }
      }
    }
    for (auto &a : node_map) {
      if (a.first == -1) {
        delete a.second;
        continue;
      }
      if (!a.second->data) {
        delete a.second->val;
      } else {
        delete a.second->data;
      }
      delete a.second;
    }
    delete[] matrices;

}