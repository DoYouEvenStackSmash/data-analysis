#!/usr/bin/python3
from clustering_driver import *
from clustering_imports import *

from decimal import Decimal


def original_likelihood(omega, m, N_pix, noise=1):
    """
    Likelihood function from the paper
    L(\omega, m, N_{\text{pix}}, \text{noise}) = \frac{1}{{(2 \cdot \text{noise}^2 \cdot \pi)^{\frac{N_{\text{pix}}}{2}}}} \cdot \exp\left(-\frac{{\|\omega - m\|^2}}{{2 \cdot \text{noise}^2}}\right)

    """
    L = np.power(
        2.0 * np.square(noise) * np.pi, -1.0 * (np.divide(N_pix, 2.0))
    ) * np.exp(
        -1.0
        * np.divide(np.square(np.linalg.norm(omega - m)), (2.0 * (np.square(noise))))
    )
    return L


def postprocessing_adjust(input_arr):
    """
    Placeholder for modifying an array
    """
    return [np.log(i) for i in input_arr]


def likelihood(omega, m, N_pix, noise=1):
    """
    New likelihood
    """
    lambda_square = noise**2
    coeff = 1 / np.sqrt(2 * np.pi * (lambda_square))
    l2 = np.exp(-1.0 * (np.square(np.linalg.norm(omega - m)) / (2 * lambda_square)))
    return coeff * l2


def evaluate_tree_neighbor_likelihood(node_list, data_list, input_list, noise=1):
    """
    Given a hierarchical clustering of the images_from_structures, and the input images,
    evaluate the likelihood of each image with its nearest structure

    Nearest neighbor search
    returns a list of likelihoods associated with the data_list members
    """
    # in application, this is 16384 <- 128 x 128
    n_pix = data_list[0].shape[1] ** 2.0

    # Likelihood array
    likelihood_omega_m = [0.0 for _ in range(len(input_list))]

    start_time = time.perf_counter()

    # omega := experimental image
    for input_index, omega in enumerate(input_list):
        nearest_index, nearest_distance = search_tree(node_list, data_list, omega)
        likelihood_omega_m[input_index] += likelihood(
            omega, data_list[nearest_index], n_pix, noise
        )

    end_time = time.perf_counter() - start_time
    logger.info("tree_match_likelihood time: {}".format(end_time))

    likelihood_omega_m = postprocessing_adjust(likelihood_omega_m)
    return likelihood_omega_m


def evaluate_tree_cluster_likelihood(node_list, data_list, input_list, noise=1):
    """
    Given a hierarchical clustering of the images_from_structures, and the input images,
    Accumulate the likelihoods calculated between each image and the cluster of structures it is associated with

    returns a list of likelihoods associated with the data_list members
    """
    n_pix = data_list[0].shape[1] ** 2

    likelihood_omega_m = [0.0 for _ in range(len(input_list))]  # Likelihood array

    start_time = time.perf_counter()

    # omega := experimental image
    # for each omega, find the cluster it belongs to and calculate the likelihoods of it with all members of the cluster
    for input_index, omega in enumerate(input_list):
        cluster_node = node_list[find_cluster(node_list, omega)]
        for data_idx in cluster_node.data_refs:
            likelihood_omega_m[input_index] += likelihood(
                omega, data_list[data_idx], n_pix, noise
            )

    end_time = time.perf_counter() - start_time
    logger.info("tree_cluster_likelihood time: {}".format(end_time))

    likelihood_omega_m = postprocessing_adjust(likelihood_omega_m)
    return likelihood_omega_m


def calculate_noise(input_list):
    """
    Calculate the noise as the standard deviation
    """
    avg = np.mean(input_list)
    noise = np.sqrt(
        np.divide(
            np.sum(np.array([np.square(x - avg) for x in input_list])),
            input_list.shape[0] - 1,
        )
    )
    return noise


def search_tree_likelihoods(node_list, data_list, input_list, input_noise=None):
    """
    Given a hierarchical clustering of the images_from_structures, and the input images,
    Accumulate the likelihood according to the paper in the structure indices

    Two flavors: Nearest neighbor, and cluster membership likelihoods
    Returns nearest neighbor likelihoods and cluster likelihoods
    """
    if input_noise is None:
        input_noise = calculate_noise(input_list)

    nn_likelihoods = evaluate_tree_neighbor_likelihood(
        node_list, data_list, input_list, input_noise
    )
    cluster_likelihoods = evaluate_tree_cluster_likelihood(
        node_list, data_list, input_list, input_noise
    )
    logger.info("lambda: {}".format(input_noise))
    return nn_likelihoods, cluster_likelihoods


def evaluate_global_neighbor_likelihood(data_list, input_list, noise=1):
    """
    Given reference data and some input data,
    Accumulate the likelihoods according to the paper in the structure indices

    Nearest neighbor
    returns a list of likelihoods associated with the data_list members
    """
    n_pix = data_list[0].shape[0] * data_list[0].shape[1]
    likelihood_omega_m = [0.0 for _ in range(len(input_list))]
    start_time = time.perf_counter()

    for idx, omega in enumerate(input_list):
        min_distance = float("inf")
        nn_index = 0
        for midx, m in enumerate(data_list):
            distance = np.linalg.norm(omega - m)
            # Update nearest neighbor if a closer one is found
            if distance < min_distance:
                min_distance = distance
                nn_index = midx
        # for i in range(n_pix):
        likelihood_omega_m[idx] += likelihood(omega, m, n_pix, noise)

    end_time = time.perf_counter() - start_time
    logger.info("global_neighbor_likelihood time: {}".format(end_time))

    likelihood_omega_m = postprocessing_adjust(likelihood_omega_m)
    return likelihood_omega_m


def evaluate_global_likelihood(data_list, input_list, noise=1):
    """
    Given reference data and some input data,
    Accumulate the likelihoods according to the paper in the structure indices

    """
    n_pix = data_list[0].shape[1] ** 2.0
    likelihood_omega_m = [0.0 for _ in range(len(input_list))]
    start_time = time.perf_counter()

    for idx, omega in enumerate(input_list):
        for midx, m in enumerate(data_list):
            likelihood_omega_m[idx] += likelihood(omega, m, n_pix, noise)

    end_time = time.perf_counter() - start_time
    logger.info("global_likelihood time: {}".format(end_time))

    likelihood_omega_m = postprocessing_adjust(likelihood_omega_m)
    return likelihood_omega_m


def global_scope_likelihoods(data_list, input_list, input_noise=None):
    """
    Given reference data and some input data,
    Returns nearest neighbor likelihoods and all pairs likelihoods likelihoods
    """
    if input_noise is None:
        input_noise = calculate_noise(input_list)

    nn_likelihoods = evaluate_global_neighbor_likelihood(
        data_list, input_list, input_noise
    )
    global_likelihoods = evaluate_global_likelihood(data_list, input_list, input_noise)
    logger.info("lambda: {}".format(input_noise))
    return nn_likelihoods, global_likelihoods


def write_csv(single_point_likelihood, area_likelihood, filename="out.csv"):
    f = open(filename, "w")

    head = f"single_point_likelihood, area_likelihood,\n"
    f.write(head)
    for i, sp in enumerate(single_point_likelihood):
        f.write(f"{sp},{area_likelihood[i]},\n")
    f.close()
