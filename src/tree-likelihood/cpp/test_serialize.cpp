#include <iostream>
#include <fstream>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include "kmedioids.h"
#include "kmeans.h"
#include "tree_builder.h"
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>

#include <ctime>
#include <cstdlib>
// #include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <chrono>
// #include "CTNode.h"
#include "aux_functions.h"
#include "tree_serializer.h"
#include <bits/stdc++.h>



int main() {
  vector<MatrixXd> data_store;
  vector<int> data_refs;
  
  mock_data_loader(data_store, data_refs);
  map<int,CTNode*> node_map = construct_tree(data_store.data(), data_refs, 5, 15, 100);

  // Create a JSON document
  Document document;
  document.SetObject();

  postprocess_node_map(node_map, data_store.data(), document);

  StringBuffer buffer;
  Writer<StringBuffer> writer(buffer);
  document.Accept(writer);
  std::cout << buffer.GetString() << std::endl;
  
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
  return 0;
}