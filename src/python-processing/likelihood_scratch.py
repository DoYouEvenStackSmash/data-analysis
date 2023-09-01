#!/usr/bin/python3
from clustering_driver import *
from clustering_imports import *


def _likelihood(omega, m, N_pix, noise=1):
    """
    Likelihood function from the paper
    """
    L = np.power(
        2.0 * np.square(noise) * np.pi, -1.0 * (np.divide(N_pix, 2.0))
    ) * np.exp(
        -1.0
        * np.divide(np.square(np.linalg.norm(omega - m)), (2.0 * (np.square(noise))))
    )

    return L


def likelihood(omega, m, N_pix, noise=1):
    """
    Same likelihood but from a different resource, formatted differently.
    Not sure what the procedure should be for all images
    """
    L = np.divide(
        np.exp(
            -1.0
            * np.divide(
                np.square(np.linalg.norm(omega - m)), (2.0 * (np.square(noise)))
            )
        ),
        np.power(2.0 * np.square(noise) * np.pi, (np.divide(N_pix, 2.0))),
    )
    return L


def evaluate_tree_neighbor_likelihood(node_list, data_list, input_list):
    """
    Given a hierarchical clustering of the images_from_structures, and the input images,
    evaluate the likelihood of each image with its nearest structure

    Nearest neighbor search
    returns a list of likelihoods associated with the data_list members
    """
    # in application, this is 16384 <- 128 x 128
    n_pix = np.product(data_list[0].shape[1:])

    # Likelihood array
    likelihood_omega_m = [0.0 for _ in range(len(data_list))]

    start_time = time.perf_counter()

    # omega := experimental image
    for input_index, omega in enumerate(input_list):
        nearest_index, nearest_distance = search_tree(node_list, data_list, omega)
        likelihood_omega_m[nearest_index] += likelihood(
            omega, data_list[nearest_index], n_pix
        )

    end_time = time.perf_counter() - start_time
    logger.info("tree_match_likelihood time: {}".format(end_time))

    return likelihood_omega_m


def evaluate_tree_cluster_likelihood(node_list, data_list, input_list):
    """
    Given a hierarchical clustering of the images_from_structures, and the input images,
    Accumulate the likelihoods calculated between each image and the cluster of structures it is associated with

    returns a list of likelihoods associated with the data_list members
    """
    n_pix = np.product(data_list[0].shape[1:])

    likelihood_omega_m = [0.0 for _ in range(len(data_list))]  # Likelihood array

    start_time = time.perf_counter()

    # omega := experimental image
    # for each omega, find the cluster it belongs to and calculate the likelihoods of it with all members of the cluster
    for input_index, omega in enumerate(input_list):
        cluster_node = node_list[find_cluster(node_list, omega)]
        for data_idx in cluster_node.data_refs:
            likelihood_omega_m[data_idx] += likelihood(
                omega, data_list[data_idx], n_pix
            )

    end_time = time.perf_counter() - start_time
    logger.info("tree_cluster_likelihood time: {}".format(end_time))
    return likelihood_omega_m


def search_tree_likelihoods(node_list, data_list, input_list):
    """
    Given a hierarchical clustering of the images_from_structures, and the input images,
    Accumulate the likelihood according to the paper in the structure indices

    Two flavors: Nearest neighbor, and cluster membership likelihoods
    Returns nearest neighbor likelihoods and cluster likelihoods
    """
    nn_likelihoods = evaluate_tree_neighbor_likelihood(node_list, data_list, input_list)
    cluster_likelihoods = evaluate_tree_cluster_likelihood(
        node_list, data_list, input_list
    )
    return nn_likelihoods, cluster_likelihoods


def evaluate_global_neighbor_likelihood(data_list, input_list):
    """
    Given reference data and some input data,
    Accumulate the likelihoods according to the paper in the structure indices

    Nearest neighbor
    returns a list of likelihoods associated with the data_list members
    """
    n_pix = np.product(data_list[0].shape[1:])
    likelihood_omega_m = [0.0 for _ in range(len(data_list))]
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
        likelihood_omega_m[nn_index] += likelihood(omega, m, n_pix)

    end_time = time.perf_counter() - start_time
    logger.info("global_neighbor_likelihood time: {}".format(end_time))
    return likelihood_omega_m


def evaluate_global_likelihood(data_list, input_list):
    """
    Given reference data and some input data,
    Accumulate the likelihoods according to the paper in the structure indices

    """
    n_pix = np.product(data_list[0].shape[1:])
    likelihood_omega_m = [0.0 for _ in range(len(data_list))]
    start_time = time.perf_counter()

    for idx, omega in enumerate(input_list):
        for midx, m in enumerate(data_list):
            likelihood_omega_m[midx] += likelihood(omega, m, n_pix)

    end_time = time.perf_counter() - start_time
    logger.info("global_likelihood time: {}".format(end_time))

    return likelihood_omega_m


def global_scope_likelihoods(data_list, input_list):
    """
    Given reference data and some input data,
    Returns nearest neighbor likelihoods and all pairs likelihoods likelihoods
    """
    nn_likelihoods = evaluate_global_neighbor_likelihood(data_list, input_list)
    global_likelihoods = evaluate_global_likelihood(data_list, input_list)
    return nn_likelihoods, global_likelihoods


def write_csv(single_point_likelihood, area_likelihood, filename="out.csv"):
    f = open(filename, "w")

    head = f"single_point_likelihood, area_likelihood,\n"
    f.write(head)
    for i, sp in enumerate(single_point_likelihood):
        f.write(f"{sp},{area_likelihood[i]},\n")
    f.close()
