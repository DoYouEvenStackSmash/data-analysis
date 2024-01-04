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
#include <bits/stdc++.h>
using namespace std;
// #include <nlohmann/json.hpp>

// using json = nlohmann::json;
using namespace rapidjson;

void serializeData(Document& document, map<int,CTNode*> node_map) {
    // Create the JSON structure
    Value parameters(kObjectType);
    parameters.AddMember("input", "your_input_value", document.GetAllocator());
    parameters.AddMember("output", "your_output_value", document.GetAllocator());
    // Add other parameters as needed

    Value resources(kObjectType);
    resources.AddMember("node_vals_file", "your_node_vals_file_value", document.GetAllocator());
    resources.AddMember("data_list_file", "your_data_list_file_value", document.GetAllocator());
    resources.AddMember("param_list_file", "your_param_list_file_value", document.GetAllocator());
    // Add other resource fields as needed

    Value nodeList(kArrayType);
    
    std::string mid_split = ",";//" -> ";
    for (auto &a : node_map) {
      Value node(kObjectType);
      if (a.first != -1 && a.second->data != nullptr) {
        node.AddMember("node_id",0,document.GetAllocator());
        
        for (auto &b : (*a.second->data))
          cout << "N"<< a.first << "" << mid_split << "D" << b << "\n";
      }else {
        for (auto &b : a.second->children) {
          cout << "N"<< a.first << ""<< mid_split <<"N" << b << "\n";
        }
      }
    }
    Value node1(kObjectType);
    node1.AddMember("node_id", 0, document.GetAllocator());
    // Add other fields of node1 as needed

    Value node2(kObjectType);
    node2.AddMember("node_id", 1, document.GetAllocator());
    // Add other fields of node2 as needed

    nodeList.PushBack(node1, document.GetAllocator());
    nodeList.PushBack(node2, document.GetAllocator());

    // Add the created values to the document
    document.AddMember("parameters", parameters, document.GetAllocator());
    document.AddMember("resources", resources, document.GetAllocator());
    document.AddMember("node_list", nodeList, document.GetAllocator());
}

int main() {
    vector<MatrixXd> data_store;
    vector<int> data_refs;
    
    mock_data_loader(data_store, data_refs);
    // return 0;
    // int C = data_refs.size();
    auto start = std::chrono::steady_clock::now();
    map<int,CTNode*> node_map = construct_tree(data_store.data(), data_refs, 3, 10, 100);
    cout <<"done";
    auto end = std::chrono::steady_clock::now();
    auto time_diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    // // Create a JSON document
    // Document document;
    // document.SetObject();

    // // Serialize data into the document
    // serializeData(document,node_map);

    // // Output the JSON as a string
    // StringBuffer buffer;
    // Writer<StringBuffer> writer(buffer);
    // document.Accept(writer);

    // // Print or save the JSON string as needed
    // std::cout << buffer.GetString() << std::endl;

    // // Save the JSON to a file (replace "output.json" with the desired file path)
    // std::ofstream outputFile("output.json");
    // outputFile << buffer.GetString();
    // outputFile.close();
    
    CTNode* root = node_map[1];
    
    cout << root->_id << endl;
    for (auto &c : root->children) {
      cout << c << ',';
    }
    // return 0;
    
    std::string mid_split = ",";//" -> ";
    
    // for (auto &a : node_map) {
    //   if (a.first != -1 && a.second->data != nullptr) {
    //     for (auto &b : (*a.second->data))
    //       cout << "N"<< a.first << "" << mid_split << "D" << b << "\n";
    //   }else {
    //     for (auto &b : a.second->children) {
    //       cout << "N"<< a.first << ""<< mid_split <<"N" << b << "\n";
    //     }
    //   }
    // }
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
    return 0;
}