#!/usr/bin/python3

import csv
import numpy as np
import scipy.io
import argparse


def matlab_convert(filename, data_dict):
    scipy.io.savemat(f"analytics_data.mat", data_dict)


def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return left


def find_index_in_naive(naive_matrix, prefix, target_vector, output_dict):
    sorted_arr = np.sort(naive_matrix, axis=1)
    print(sorted_arr)
    output_dict[f"{prefix}_elem_indices"] = []
    for i, row in enumerate(sorted_arr):
        output_dict[f"{prefix}_elem_indices"].append(
            binary_search(row, target_vector[i])
        )
    output_dict[f"{prefix}_elem_indices"] = np.array(
        output_dict[f"{prefix}_elem_indices"]
    )
    return output_dict


def read_csv(filename):
    columns = {}
    with open(filename, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)
        header.remove("")

        for i, col in enumerate(header):
            columns[col] = []

        for row in csv_reader:
            columns[header[0]].append(float(row[0]))
            columns[header[1]].append(float(row[1]))

    for col in columns:
        columns[col] = np.array(columns[col])
    return columns


def load_naive_matrix(filename, output_dict):
    output_dict["naive_matrix"] = np.load(filename)


def main():
    parser = argparse.ArgumentParser(
        description="Read CSV and load NumPy array from specified files."
    )

    parser.add_argument("-t", "--tree_likelihoods", help="CSV filename (A.csv)")
    parser.add_argument(
        "-n", "--naive_likelihoods", help="NumPy array filename (b.npy)"
    )

    args = parser.parse_args()

    csv_data = read_csv(args.tree_likelihoods)
    output_dict = {}
    output_dict.update(csv_data)
    load_naive_matrix(args.naive_likelihoods, output_dict)
    naive_matrix = output_dict["naive_matrix"]
    prefixes = ["single_point", "area"]
    output_dict = find_index_in_naive(
        naive_matrix, prefixes[0], csv_data["single_point_likelihood"], output_dict
    )
    output_dict = find_index_in_naive(
        naive_matrix, prefixes[1], csv_data["area_likelihood"], output_dict
    )
    print(output_dict["area_elem_indices"])
    matlab_convert("foo", output_dict)


if __name__ == "__main__":
    main()
