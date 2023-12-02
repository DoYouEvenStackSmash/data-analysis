#include <iostream>
#include <eigen3/Eigen/Dense>
#include <map>
// #include <vector>
#include <eigen3/Eigen/StdVector>

using Eigen::MatrixXd;
 
struct CTNode {
  MatrixXd* val = nullptr;
  // int _id = -1;
  // std::vector<int> children;
  // std::vector<int>* data = nullptr;
};

MatrixXd* init_single_matrix() {
  MatrixXd* m = new MatrixXd(2,2);
  // m->setZero();
  (*m)(0,0) = 3;
  (*m)(1,0) = 2.5;
  (*m)(0,1) = -1;
  (*m)(1,1) = 0;
  return m;
}

// void nested_vector_test() {


void initialize_struct_map() {
  std::map<int, CTNode*> nmap;
  for (int i = 0; i < 10; i++) {
    nmap[i] = nullptr;
    MatrixXd *m = init_single_matrix();
    CTNode* new_node = new CTNode;
    new_node->val = m;
    nmap[i] = new_node;
  }

  for (auto a : nmap) {
    printf("%i\t", a.first);
    delete a.second->val;
    delete a.second;
  }
}

void initialize_vector_matrix() {
  // initialize vector of matrix pointers
  std::vector<MatrixXd*> foo;

  for (int i = 0; i < 10; i++) {
    MatrixXd *m = init_single_matrix();
    foo.push_back(m);
  }

  printf("%f", (*foo[2])(0,0));
  
  // delete all the initialized matrices
  for (auto a : foo) {
    // std::cout << typeid(a).name() << std::endl;
    delete a;
  }
  // std::cout << foo.size() << std::endl;
}

void init_vector_stack() {
  std::vector<MatrixXd> foo;
  int i = 0;
  for (i = 0; i < 10000; i++) {
    MatrixXd x(2,2);
    x(0,0) = i;
    x(0,1) = i + 1;
    x(1,0) = i - 1;
    x(1, 1) = i;
    foo.push_back(x);
  }
  for (auto &a : foo) {
    printf("%f %f\n%f %f\n\n", a(0,0), a(0,1),a(1,0), a(1,1));
  }
}
int main()
{
  init_vector_stack();
  // initialize_vector_matrix();
  // initialize_struct_map();
  MatrixXd m(2,2);
  // m.setZero();

  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = 0;//m(1,0) + m(0,1);
  printf("%f", m(0,0));
  // std::cout << m << std::endl;
}