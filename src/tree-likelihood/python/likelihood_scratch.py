#!/usr/bin/python3
from clustering_driver import *
from clustering_imports import *
from tree_search_operations import *
from greedy_traversal import *
from decimal import Decimal
from patient_traversal import *
from bounded_traversal import *
from likelihood_helpers import *


def testbench_likelihood(node_list, data_list, input_list, input_noise=None):
    """
    Testbench function for head to head comparison of tree approximation and global search
    """

    nn_likelihoods = bounded_tree_likelihood(node_list, data_list, input_list)
    global_likelihoods = patient_tree_likelihood(node_list, data_list, input_list)
    # naive_likelihoods = alt_naive_likelihood(node_list, data_list, input_list)
    # np.save("naive_likelihoods.npy", naive_likelihoods)
    # sys.exit()
    return nn_likelihoods, global_likelihoods


def write_csv(single_point_likelihood, area_likelihood, filename="out.csv"):
    f = open(filename, "w")

    head = f"single_point_likelihood,area_likelihood,\n"
    f.write(head)
    for i, sp in enumerate(single_point_likelihood):
        f.write(f"{sp},{area_likelihood[i]},\n")
    f.close()


def compare_tree_likelihoods(node_list, data_list, input_list):
    bounded_likelihood_mat = np.array(
        _unbounded_tree_likelihood(node_list, data_list, input_list)
    )
    dist_mat = np.zeros(
        (bounded_likelihood_mat.shape[0], bounded_likelihood_mat.shape[1])
    )
    for i, row in enumerate(bounded_likelihood_mat):
        for j, col in enumerate(row):
            dist_mat[i][int(col[1])] = col[0]
    np.save("dist_mat.npy", dist_mat)
    naive_likelihood_mat = alt_naive_likelihood(node_list, data_list, input_list)
    np.save("naive_likelihoods.npy", naive_likelihood_mat)
    sys.exit()


def alt_naive_likelihood(node_list, data_list, input_list):
    dist_mat = np.zeros((len(input_list), len(data_list)))

    # minst = np.zeros((len(input_list)), dtype=np.float32)
    # ld = len(data_list)
    noise = calculate_noise(input_list)
    print(noise)
    lambda_square = noise**2
    start_time = time.perf_counter()
    for i, T in enumerate(input_list):
        min_dist = float("Inf")
        for j, d in enumerate(data_list):
            # dist_mat[i,j] = original_likelihood(T.m1,d.m1, 1, noise )

            #
            dist_mat[i, j] = np.exp(
                -1.0
                * (
                    (difference_calculation(T.m1, d.m1, noise) ** 2)
                    / (2 * lambda_square)
                )
            )
            # print(f.shape)
        dist_mat[i] = postprocessing_adjust(dist_mat[i], noise, 1)
        print(i)

    end_time = time.perf_counter() - start_time

    LOGGER.info("{},{}".format(len(input_list) ** 2, end_time))
    return dist_mat


def _alt_naive_likelihood(node_list, data_list, input_list):
    input_set = np.array([input_list[r].m1 for r in range(len(input_list))])
    data_set = np.array([data_list[r].m1 for r in range(len(data_list))])
    dist_mat = np.zeros((len(input_list), len(data_list)))
    noise = calculate_noise(input_list)
    # print(noise)
    lambda_square = noise**2
    for x in range(len(input_set)):
        print(x)
        dist_mat[x] = np.exp(
            -1.0
            * (
                np.sum(np.square((input_set[x] - data_set) / noise), axis=(-2, -1))
                / (2 * lambda_square)
            )
        )

    dist_mat = postprocessing_adjust(dist_mat, noise, 1)
    return dist_mat
