#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <limits>
#include "kmeans.h"
#include <set>
using namespace std;

void params(bool foo=false) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double sample_val = dist(gen);
    cout << "sample:\t" << sample_val << endl;
}

int main() {
  // for (int i = 0; i < 20; ++i){
  //   params();
  // }
  // return 0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distribution(0, 10); // Generates integers between 1 and 100
    int rand_start = distribution(gen);
    set<int> s;
  //   for (int i = 0; i < 10; i++) {
  //     if (i == rand_start)
  //       continue;
  //     s.insert(i);
  //   }

  // for (auto &a : s)
  //   cout << a << endl;
  // return 0;
  vector<vector<int>> c(10);
    Eigen::MatrixXf matrix1(2, 2);
    matrix1 << 1.0, 2.0,
               3.0, 4.0;

    Eigen::MatrixXf matrix2(2, 2);
    matrix2 << 2.0, 2.0,
               3.0, 4.0;
    cout << (matrix1 - matrix2).norm() << endl;
  
  // for (auto a: c)
  //   cout << a.size() << "\n";
  vector<Eigen::MatrixXf> cent = {matrix1, matrix2};
  int dsize = 6;
  Eigen::MatrixXf data[dsize];
  for (int i = 0; i < dsize; ++i) {
    data[i].resize(2,2);
  }
  data[0] = matrix1;
  data[1] = matrix2;
  data[2] = matrix1+ matrix2;
  data[3] = matrix1 * matrix2;
  data[4] = data[3] * data[3];
  data[5] = data[0] + data[0];

  // cout << data[0] << end;
  vector<int> data_refs = {0,1,2,3,4,5};
  vector<Eigen::MatrixXf> centroids = {matrix1, matrix2};
  vector<Eigen::MatrixXf> nctr;
  vector<vector<int>> nrc;
  // kmeans_refs(&data[0], data_refs, centroids, true, nctr, nrc);
  int starting_index = 2;
  int k = 2;
  kmeanspp_refs(&data[0], data_refs, starting_index, k, nctr);
  cout << "centers" << endl;
  for (auto a: nctr) {
    // for (auto b : a)
      cout << a;
      cout << endl;
  }
  return 0;
  Eigen::MatrixXf mean_mat(2,2);
  // mean_mat << 0.0
  mean_mat.setZero();
  // mean_mat.resize(2,2);
  for (int i = 0; i < cent.size(); ++i) {
    mean_mat = mean_mat + cent[i];
  }
  mean_mat /=cent.size();
  cout << mean_mat << endl;
  cout << typeid(cent.data()).name() << endl;
  params();  
  double infinity = numeric_limits<double>::infinity();
  // cout << (infinity > 1) << endl;
  return 0;
}