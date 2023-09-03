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

def likelihood(omega, m, N_pix, noise = 1, slice_size=128):
    
    # npix = omega.shape[0] * omega.shape[1]
    L = 0.0
    denom = (2.0 * np.square(noise) * np.pi) ** (np.divide(slice_size, 2.0))
    for i in range(omega.shape[0]):
        for j in range(0, omega.shape[1], slice_size):
            l2 = np.square(np.linalg.norm(omega[i, j : j + slice_size] - m[i, j:j+slice_size]))
            L += np.exp(-1.0* np.divide(l2, (2.0 * (np.square(noise))))) / denom
    return L


def __likelihood(omega, m, N_pix, noise=1):#0.2727643646373728):#0.07440039861602968):
    """
    Same likelihood but from a different resource, formatted differently.
    Not sure what the procedure should be for all images
    """
    l2 = np.square(np.linalg.norm(omega - m))
    # overflow!
    denom = (2.0 * np.square(noise) * np.pi) ** (np.divide(N_pix, 2.0))
    
    # denom = 1

    L = np.exp(-1.0* np.divide(l2, (2.0 * (np.square(noise)))))
    # print(denom)
    return L / denom


def evaluate_tree_neighbor_likelihood(node_list, data_list, input_list, noise = 1):
    """
    Given a hierarchical clustering of the images_from_structures, and the input images,
    evaluate the likelihood of each image with its nearest structure

    Nearest neighbor search
    returns a list of likelihoods associated with the data_list members
    """
    # in application, this is 16384 <- 128 x 128
    n_pix = data_list[0].shape[1] ** 2.0

    # Likelihood array
    likelihood_omega_m = [0.0 for _ in range(len(data_list))]

    start_time = time.perf_counter()

    # omega := experimental image
    for input_index, omega in enumerate(input_list):
        nearest_index, nearest_distance = search_tree(node_list, data_list, omega)
        likelihood_omega_m[nearest_index] += likelihood(
            omega, data_list[nearest_index], n_pix, noise
        )

    end_time = time.perf_counter() - start_time
    logger.info("tree_match_likelihood time: {}".format(end_time))

    return likelihood_omega_m


def evaluate_tree_cluster_likelihood(node_list, data_list, input_list, noise = 1):
    """
    Given a hierarchical clustering of the images_from_structures, and the input images,
    Accumulate the likelihoods calculated between each image and the cluster of structures it is associated with

    returns a list of likelihoods associated with the data_list members
    """
    n_pix = data_list[0].shape[1] ** 2

    likelihood_omega_m = [0.0 for _ in range(len(data_list))]  # Likelihood array

    start_time = time.perf_counter()

    # omega := experimental image
    # for each omega, find the cluster it belongs to and calculate the likelihoods of it with all members of the cluster
    for input_index, omega in enumerate(input_list):
        cluster_node = node_list[find_cluster(node_list, omega)]
        for data_idx in cluster_node.data_refs:
            likelihood_omega_m[data_idx] += likelihood(
                omega, data_list[data_idx], n_pix, noise
            )

    end_time = time.perf_counter() - start_time
    logger.info("tree_cluster_likelihood time: {}".format(end_time))
    return likelihood_omega_m

def calculate_noise(input_list):
    """
    Calculate the noise as the standard deviation
    """
    avg = np.mean(input_list)
    noise = np.sqrt(np.divide(np.sum(np.array([np.square(x - avg) for x in input_list])), input_list.shape[0] - 1))
    return noise
    

def search_tree_likelihoods(node_list, data_list, input_list, input_noise = None):
    """
    Given a hierarchical clustering of the images_from_structures, and the input images,
    Accumulate the likelihood according to the paper in the structure indices

    Two flavors: Nearest neighbor, and cluster membership likelihoods
    Returns nearest neighbor likelihoods and cluster likelihoods
    """
    if input_noise is None:
        input_noise = calculate_noise(input_list)

    nn_likelihoods = evaluate_tree_neighbor_likelihood(node_list, data_list, input_list, input_noise)
    cluster_likelihoods = evaluate_tree_cluster_likelihood(
        node_list, data_list, input_list, input_noise
    )
    logger.info("lambda: {}".format(input_noise))
    return nn_likelihoods, cluster_likelihoods


def evaluate_global_neighbor_likelihood(data_list, input_list, noise = 1):
    """
    Given reference data and some input data,
    Accumulate the likelihoods according to the paper in the structure indices

    Nearest neighbor
    returns a list of likelihoods associated with the data_list members
    """
    n_pix = data_list[0].shape[0] * data_list[0].shape[1]
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
        # for i in range(n_pix):
        likelihood_omega_m[nn_index] += likelihood(omega, m, n_pix, noise)

    end_time = time.perf_counter() - start_time
    logger.info("global_neighbor_likelihood time: {}".format(end_time))
    return likelihood_omega_m


def evaluate_global_likelihood(data_list, input_list, noise = 1):
    """
    Given reference data and some input data,
    Accumulate the likelihoods according to the paper in the structure indices

    """
    n_pix = data_list[0].shape[1] ** 2.0
    likelihood_omega_m = [0.0 for _ in range(len(data_list))]
    start_time = time.perf_counter()

    for idx, omega in enumerate(input_list):
        for midx, m in enumerate(data_list):
            likelihood_omega_m[midx] += likelihood(omega, m, n_pix, noise)

    end_time = time.perf_counter() - start_time
    logger.info("global_likelihood time: {}".format(end_time))

    return likelihood_omega_m


def global_scope_likelihoods(data_list, input_list, input_noise = None):
    """
    Given reference data and some input data,
    Returns nearest neighbor likelihoods and all pairs likelihoods likelihoods
    """
    if input_noise is None:
        input_noise = calculate_noise(input_list)

    nn_likelihoods = evaluate_global_neighbor_likelihood(data_list, input_list, input_noise)
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
