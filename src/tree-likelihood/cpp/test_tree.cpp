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
#include "aux_functions.h"

int main() {
    const int numMatrices = 20;
    const int numRows = 1;
    const int numCols = 2;

    vector<MatrixXd> data_store;
    vector<int> data_refs;
    
    mock_data_loader(data_store, data_refs);
    // return 0;
    // int C = data_refs.size();
    auto start = std::chrono::steady_clock::now();
    map<int,CTNode*> node_map = construct_tree(data_store.data(), data_refs, 8, 10, 100);
    cout <<"done";
    auto end = std::chrono::steady_clock::now();
    auto time_diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "time:\t" << time_diff / 1e9f << "\n";//<< "total:\t" << total << "\n";

    std::string mid_split = ",";//" -> ";
    
    for (auto &a : node_map) {
      if (a.first != -1 && a.second->data != nullptr) {
        for (auto &b : (*a.second->data))
          cout << "N"<< a.first << "" << mid_split << "D" << b << "\n";
      }else {
        for (auto &b : a.second->children) {
          cout << "N"<< a.first << ""<< mid_split <<"N" << b << "\n";
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
    // delete[] matrices;

}