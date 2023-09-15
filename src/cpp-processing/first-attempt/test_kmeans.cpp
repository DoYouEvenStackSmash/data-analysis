#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <ctime>
#include <cstdlib>

#include <fstream>
#include <vector>
#include <sstream>
#include <iomanip>

#include "kmeans.h"
#include "aux_functions.h"
using Eigen::MatrixXd;

int main(void) {


    vector<MatrixXd> data_store;
    vector<int> data_refs;
    mock_data_loader(data_store,data_refs);
    vector<MatrixXd> centroids;
    int starting_index = 2;
    int k = 12;
    kmeanspp_refs(data_store.data(), data_refs, starting_index, k, centroids);
    vector<vector<int>> ref_clusters;
    for (int i = 0; i < 100; ++i) {
      vector<MatrixXd> new_ctr;
      vector<vector<int>> new_cls;
      kmeans_refs(data_store.data(), data_refs, centroids, true, new_ctr, new_cls);
      ref_clusters = new_cls;
      // ref_clusters = new_cls;
      if (new_ctr.size() < centroids.size()) {
        centroids = new_ctr;
        break;
      }
      centroids = new_ctr;
    }
    // return 0;
    std::vector<std::vector<std::vector<double>>> data(ref_clusters.size());
    std::vector<std::vector<double>> data_centroids(ref_clusters.size());
    // for (int i = 0; i < ref_clusters.size(); )
    for (int i = 0; i < ref_clusters.size(); ++i) {
      int cidx = i;
      const Eigen::Map<const Eigen::VectorXd> flattened_centroid(centroids[cidx].data(), centroids[cidx].size());
      data_centroids[i].assign(flattened_centroid.data(), flattened_centroid.data() + flattened_centroid.size());
      for (int j = 0; j < ref_clusters[i].size(); ++j) {
        int idx = ref_clusters[i][j];
        const Eigen::Map<const Eigen::VectorXd> flattened_vector(data_store[idx].data(), data_store[idx].size());
        std::vector<double> vec;
        vec.assign(flattened_vector.data(), flattened_vector.data() + flattened_vector.size());
        data[i].push_back(vec);
      }
    }

    serialize_clusters(data, data_centroids);
    return 0;
}

