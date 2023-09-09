#include <iostream>
#include <eigen3/Eigen/Dense>
#include <random>
#include <chrono>
#include <cmath>
int main() {
    // Define the size of each matrix and the number of matrices
    const int matrix_size = 128;
    const int num_matrices = 512 * 2;
    const int num_images = 8192 * 3;

    // Create an array of Eigen matrices
    Eigen::MatrixXf *models = new Eigen::MatrixXf[num_matrices];
    Eigen::MatrixXf *images = new Eigen::MatrixXf[num_images];

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 100.0);


    // Initialize each matrix with random values
    for (int i = 0; i < num_matrices; ++i) {
        models[i].resize(matrix_size, matrix_size);  // Resize the matrix if needed
        for (int j = 0; j < matrix_size; ++j) {
            for (int k = 0; k < matrix_size; ++k) {
                models[i](j, k) = dist(gen) / 1e9;  // Initialize with random value
            }
        }
    }
    for (int i = 0; i < num_images; ++i) {
        images[i].resize(matrix_size, matrix_size);  // Resize the matrix if needed
        for (int j = 0; j < matrix_size; ++j) {
            for (int k = 0; k < matrix_size; ++k) {
                images[i](j, k) = dist(gen) / 1e9;  // Initialize with random value
            }
        }
    }

    double *dist_matrix = new double[num_matrices * num_images];
    for (int i =0; i < num_matrices * num_images; ++i) {
      dist_matrix[i] = 0.0;
    }
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < num_matrices; ++i) {
      for (int j = 0; j < num_images; ++j) {
        double norm = (models[i] - images[j]).norm();
        // std::cout << norm << "\n";
        dist_matrix[(i * num_images) + j] = exp(-norm * norm / (0.0729 * 2));
        // std::cout << "dist " << i * num_images + j << ":\t" << (models[i] - images[j]).norm();
      }
    }
    double total = 0.0;
    for (int i = 0; i < num_matrices * num_images; ++i) {
      // std::cout << dist_matrix[i] << "\n";
      total = total + log(dist_matrix[i]);
    }
    
    auto end = std::chrono::steady_clock::now();
    auto time_diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "time:\t" << time_diff / 1e9f << "\n" << "total:\t" << total << "\n";
    delete[] models;
    delete[] images;
    delete[] dist_matrix;

    return 0;
}