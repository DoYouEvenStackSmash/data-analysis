#include <iostream>
#include <eigen3/Eigen/Dense>
#include <random>

int main() {
    // Define the size and number of matrices
    const int matrix_size = 16;
    const int num_matrices = 10;

    // Create arrays A and B to hold the matrices
    Eigen::MatrixXf A[num_matrices][matrix_size];
    Eigen::MatrixXf B[num_matrices][matrix_size];
    // return 0;
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Initialize matrices A and B with random values
    for (int i = 0; i < num_matrices; ++i) {
        for (int j = 0; j < matrix_size; ++j) {
            for (int k = 0; k < matrix_size; ++k) {
                A[i][j](j, k) = dist(gen);  // Random value between 0 and 1 for A
                B[i][j](j, k) = dist(gen);  // Random value between 0 and 1 for B
            }
        }
    }
    return 0;
    // Compute and print the Frobenius norm between each pair of matrices
    for (int i = 0; i < num_matrices; ++i) {
        for (int j = 0; j < num_matrices; ++j) {
            float frobenius_norm = (A[i][j] - B[i][j]).norm();
            std::cout << "Frobenius Norm between A[" << i << "] and B[" << j << "]: " << frobenius_norm << std::endl;
        }
    }

    return 0;
}
