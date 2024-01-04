#ifndef TREE_SERIALIZER_H
#define TREE_SERIALIZER_H

#include <iostream>
#include <fstream>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
// #include "kmedioids.h"
// #include "kmeans.h"
// #include "tree_builder.h"
#include "CTNode.h"
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
// #include "aux_functions.h"
using namespace std;
// #include <bits/stdc++.h>

using namespace rapidjson;
void postprocess_node_map(map<int,CTNode*> node_map, MatrixXd *data_store, Document& document);

void postprocess_data_store(Document& document);

void postprocess_params(Document& document);

void serialize_tree(map<int,CTNode*> node_map, MatrixXd *data_store, /*SOME_TYPE params, */ Document& document);

#endif