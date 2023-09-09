#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <limits>
#include "kmeans.h"
using namespace std;

void params(bool foo=false) {
  cout << foo << endl;
}

int main() {
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
  kmeans_refs(&data[0], data_refs, centroids, true, nctr, nrc);
  cout << "centers" << endl;
  for (auto a: nrc) {
    for (auto b : a)
      cout << b << ",";
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