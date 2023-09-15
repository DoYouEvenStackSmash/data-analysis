#include <iostream>
#include <eigen3/Eigen/Dense>
#include <map>
#include <ctime>
// #include <vector>
#include <eigen3/Eigen/StdVector>

using Eigen::MatrixXd;


void ComputeDistanceMatrix( MatrixXd *data_store,
                            std::vector<int> &data_refs,
                            MatrixXd* &distance_matrix
                            ) {

  for (int i = 0; i < data_refs.size(); ++i) {
    for (int j = i + 1; j < data_refs.size(); ++j) {
      double distance = (data_store[data_refs[i]] - data_store[data_refs[j]]).norm();
      (*distance_matrix)(i, j) = distance;
      (*distance_matrix)(j, i) = distance;
    }
  }
  // distance_matrix = pairwise_distances;
}

MatrixXd* init_matrix() {
  int dims = 5;
  // initialize matrix with proper dimensions
  MatrixXd *a_matrix = new MatrixXd(dims, dims);
  
  a_matrix->setZero();
  for (int row = 0; row < dims; ++row) {
    for (int col = 0; col < dims; ++col) {
      (*a_matrix)(row, col) = static_cast<double>(std::rand()) / RAND_MAX * 100.0;
    }
  }
  return a_matrix;
}

void init_matrix_elem(std::vector<MatrixXd> &vec) {
  int dims = 5;
  MatrixXd a_matrix(dims,dims);
  for (int row = 0; row < dims; ++row) {
    for (int col = 0; col < dims; ++col) {
      a_matrix(row, col) = static_cast<double>(std::rand()) / RAND_MAX * 100.0;
    }
  }
  vec.push_back(a_matrix);
}
struct P {
  int idx = 0;
  double vSum = 0.0;
};

bool compareByVSum(P& left,P& right) {
    return left.vSum < right.vSum;
}

int main(void) {
  std::srand(12345);
  std::vector<MatrixXd> vec;
  std::vector<int> refs;
  int matrix_count = 10;
  for (int i = 0; i < matrix_count; ++i) {
    init_matrix_elem(vec);
  }
  for(int i = 0; i < vec.size(); ++i) {
    refs.push_back(i);
  }
  int C = matrix_count + 5;
  MatrixXd *dm = new MatrixXd(C,C);
  dm->setZero();
  ComputeDistanceMatrix(vec.data(), refs, dm);
  for (int i = 0; i < dm->rows(); ++i) {
    for (int j = 0; j < dm->cols(); ++j) {
      printf("%f\t", (*dm)(i,j));
    }
    printf("\n");
  }

  Eigen::VectorXd denoms = dm->rowwise().sum();
  std::vector<P> v_vals(vec.size());
  for (int j = 0; j < vec.size(); ++j) {
    // P a;
    // a.idx = j;
    v_vals[j].idx = j;
    for (int i = 0; i < vec.size(); ++i) {
      v_vals[i].vSum += (*dm)(i,j) / denoms[i];
    }
    // v_vals.push_back(a);
  }

  std::sort(v_vals.begin(), v_vals.end(), compareByVSum);
  for (int i = 0; i < v_vals.size(); ++i)
    printf("%d ", v_vals[i].idx);
  int k = 3;
  std::vector<int> medioid_indices;
  for (int i = 0; i < k; ++i) {
    medioid_indices.push_back(v_vals[i].idx);
  }

  delete dm;
  return 0;
  for (auto a : vec) {
    for (int row = 0; row < a.rows(); ++row) {
      for (int col = 0; col < a.cols(); ++col) {
        printf("%f ", a(row,col));
      }
      printf("\n");
    }
    printf("------------------");
  }
  // std::vector<MatrixXd*> data_store;
  // for (int i = 0; i < 10; i++) {
    // data_store.push_back(init_matrix());
  // }
  // MatrixXd dm;
  // ComputeDistanceMatrix(data_store.data(), )
  // for (int row = 0; row < a_matrix->rows(); ++row) {
  //   for (int col = 0; col < a_matrix->cols(); ++col) {
  //     printf("%f ", (*a_matrix)(row,col));
  //   }
  //   printf("\n");
  // }
  // delete a_matrix;
  return 0;
}