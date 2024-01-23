// #include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <limits>
#include <map>
#include <ctime>
#include <queue>
#include "kmeans.h"
#include "kmedioids.h"
#include "tree_builder.h"

using namespace std;

std::map<int, CTNode*> construct_tree(MatrixXd *data_store, vector<int> &init_data_refs, int k, int C, int R) {

  // initialize a map of nodes accessible by their id
  map<int, CTNode*> node_map;
  
  // initialize queue for accessing nodes in the map
  queue<int> node_id_queue;

  // initialize queue for accessing lists of elements in data store by their index
  queue<vector<int>> data_ref_queue;
  // Create a random number generator for random start
  std::random_device rd;
  std::mt19937 gen(rd());

  // load the data references
  data_ref_queue.push(init_data_refs);
  
  // set up root node
  int node_count = -1;
  CTNode* root = new CTNode();
  root->_id = node_count;
  node_map[node_count] = root;
  
  // add root to node_id_queue for later access in the node_map
  // increment node count
  // these operations should always occur together
  node_id_queue.push(node_count);
  node_count++;

  // initialize data refs, curr_node, node_id for pulling items off of 
  // queues in the main loop
  // vector<int> data_refs;
  CTNode* curr_node = nullptr;
  int node_id;
  MatrixXd *distance_matrix = new MatrixXd(C + 1, C + 1);
  distance_matrix->setZero();
  while (!node_id_queue.empty()) {
    // pop parent node off of node_id_queue
    node_id = node_id_queue.front();
    node_id_queue.pop();
    curr_node = node_map[node_id];

    // pop parent node's cluster data indexes off of data_ref_queue
    vector<int> data_refs = data_ref_queue.front();
    data_ref_queue.pop();
    
    // perform k means clustering if the number of elements is greater than cutoff
    if (data_refs.size() > C) {
      vector<MatrixXd> centroids;
      // choose a random start value within the range [0, data_refs.size())
      std::uniform_int_distribution<int> dist(0, data_refs.size() - 1);
      int random_start = dist(gen);
      
      // initialize centroids using kmeans++
      kmeanspp_refs(data_store, data_refs, random_start, k, centroids);

      vector<vector<int>> ref_clusters; // will contain indices to data_ref elements
      
      for (int i = 0; i < R; ++i) {
        vector<MatrixXd> new_centroids;
        vector<vector<int>> new_ref_clusters;
        // partitions data_refs into n<=k groups.
        kmeans_refs(data_store, data_refs, centroids, bool(i==0), new_centroids, new_ref_clusters);
        // save new_ref_clusters
        ref_clusters = new_ref_clusters;
        // return node_map;
        
        // it is possible to drop below the value of k
        // if this happens, we break. It can form an unbalanced tree, but
        // in practice it doesn't matter as long as R is sufficiently large
        if (new_centroids.size() < centroids.size()) {
          centroids = new_centroids;
          break;
        }
        // reset for next iteration
        centroids = new_centroids;
      }

      // Create new tree nodes, add them to the node_map
      // Add their _ids to the parent node's children vector
      // update the queues:
      // Push the data_refs partitions to the data_ref_queue 
      // push the node _id to the node_id_queue.  The node
      // will become the parent of the data_refs partition

      for (int i = 0; i < centroids.size(); ++i) {
        CTNode* newNode = new CTNode();
        newNode->_id = node_count;
        // initialize the new mean
        newNode->val = new MatrixXd(centroids[i]);
        
        // add to parent node
        curr_node->children.push_back(node_count);
        // add to node map
        node_map[node_count] = newNode; // Store the pointer in the map
        node_id_queue.push(node_count);
        data_ref_queue.push(ref_clusters[i]);
        node_count++;
      }
    
    } 
    // perform k medioids clustering if number of data_refs is below cutoff to ensure the cluster's representative
    // is a real member of the cluster. this is to avoid introducing artifacts that don't exist
    else {
      if (!data_refs.size())
        continue;
      vector<int> medioid_indices;
      vector<vector<int>> ref_clusters;
      
      // only perform k medioids if there are multiple elements in data_refs
      if (data_refs.size() > k) {
        // construct the pairwise distance matrix according to paper
        
        distance_matrix->setZero();
        preprocess(data_store, data_refs, distance_matrix, medioid_indices, k);
        int num_elements = data_refs.size();
        // vector<vector<int>> new_ref_clusters(medioid_indices.size());
        double total_sum = 0.0;
        double new_sum = 0.0;
        
        // kmedioids loop
        for (int i = 0; i < R; ++i) {
          std::vector<std::vector<int>> temp_ref_clusters;
          temp_ref_clusters = AssignClusters(distance_matrix, medioid_indices, num_elements);
          UpdateMedioids(distance_matrix, temp_ref_clusters, medioid_indices);
          CalculateSum(distance_matrix, medioid_indices, temp_ref_clusters, &new_sum);
          ref_clusters = temp_ref_clusters;
          if (new_sum == total_sum) {
            break;
          }
          // reset values
          total_sum = new_sum;
          new_sum = 0.0;
        }
      } 
      // handle case where data_refs is small enough to be a single cluster
      else if (data_refs.size() > 1) {
        
        // find the medioid of the cluster
        vector<double> D(data_refs.size(), 0.0);
        for (int i = 0; i < data_refs.size(); ++i) {
          for (int j = 0; j < data_refs.size(); ++j) {
            D[i] += (data_store[data_refs[i]] - data_store[data_refs[j]]).norm();
          }
        }
        // find index of the medioid
        int min_idx = 0;
        for (int c = 0; c < D.size(); ++c) {
          min_idx = D[min_idx] < D[c] ? min_idx : c;
        }
        medioid_indices.push_back(min_idx);
        ref_clusters.push_back({});
        for (int i = 0; i < data_refs.size(); ++i)
          ref_clusters[0].push_back(i);
      }
      // handle case where data_refs is 1
      else if (data_refs.size() == 1) {
        // special case for data_ref of size 1 to avoid edge cases
        // medioid_indices will refer to data_refs[0]
        medioid_indices.push_back(0);
        // ref_clusters will contain a single cluster, containing the index to data_refs[0]
        ref_clusters.push_back({0});
      }
      // create leaf nodes of the tree
      for (int i = 0; i < medioid_indices.size(); ++i) {
        CTNode* newNode = new CTNode();
        
        
        /**/
        newNode->_id = node_count;
        // set the newNode val pointer to the actual medioid object in data_store
        // to avoid excessively duplicating data. In the worst case we double our memory consumption,
        // so this is important.
        newNode->val = &data_store[data_refs[medioid_indices[i]]];
        
        /*
        // for serialization purposes it is essential that the index of the medioid object be stored in the 
        // leaf. we make use of existing newNode->_id and leave newNode->val as null
        newNode->_id = data_refs[medioid_indices[i]];
        */
        // set the newNode data pointer to a vector containing the indices in data_store
        // of the node's cluster members.  populate this vector with those indices
        // These will be used to directly access cluster data in node_map
        newNode->data = new vector<int>(ref_clusters[i].size(), 0);
        for (int k = 0; k < ref_clusters[i].size(); ++k) {
          (*newNode->data)[k] = data_refs[ref_clusters[i][k]];
        }

        // add new node to parent's children vector
        curr_node->children.push_back(node_count);
        
        // add new node to node_map, and increment node_count
        node_map[node_count] = newNode;
        node_count++;
      }
    }
  }
  delete distance_matrix;
  return node_map;
}