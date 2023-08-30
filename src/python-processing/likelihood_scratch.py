#!/usr/bin/python3
from clustering_driver import *
from clustering_imports import *


def find_cluster(node_list, T):
    n_curr = 0
    # search_list = []
    # search representative nodes
    while node_list[n_curr].data_refs is None:
        min_dist = float("inf")
        nn = 0
        for i in node_list[n_curr].children:
            dist = np.linalg.norm(node_list[i].val - T)
            if dist < min_dist:
                nn = i
                min_dist = dist
        # search_list.append(nn)
        n_curr = nn
    return n_curr


def likelihood(omega, m, N_pix, noise=1e-1):
    L = np.power(2.0 * (noise**2.0) * np.pi, (np.divide(-N_pix, 2.0))) * np.exp(
        -np.divide(np.square(np.linalg.norm(omega - m)), (2.0 * (noise**2.0)))
    )
    # print(L)
    return L


def search_tree_likelihoods(node_list, data_list, input_list):
    n_pix = np.product(data_list[0].shape[1:])

    # bio EM likelihood maybe
    P_M_OMEGA = [0.0 for _ in range(len(data_list))]

    # soft match
    P_omega_m = [0.0 for _ in range(len(data_list))]
    start_time = time.perf_counter()
    for input_index, omega in enumerate(input_list):
        cluster_node = node_list[find_cluster(node_list, omega)]
        for m_idx in cluster_node.data_refs:
            P_omega_m[m_idx] += likelihood(omega, data_list[m_idx], n_pix)

        m_idx, distance = search_tree(node_list, data_list, omega)
        P_M_OMEGA[m_idx] += likelihood(omega, data_list[m_idx], n_pix)

    end_time = time.perf_counter() - start_time
    logger.info("search_tree_likelihood time: {}".format(end_time))
    return P_M_OMEGA, P_omega_m


def all_pairs_likelihoods(data_list, input_list):
    P_M_OMEGA = [0.0 for _ in range(len(data_list))]
    P_omega_m = [0.0 for _ in range(len(data_list))]
    n_pix = np.product(data_list[0].shape[1:])
    start_time = time.perf_counter()
    for idx, omega in enumerate(input_list):
        min_distance = float("inf")
        nn_index = 0

        for midx, m in enumerate(data_list):
            # soft likelihood
            P_omega_m[midx] += likelihood(omega, m, n_pix)

            distance = np.linalg.norm(omega - m)
            # Update nearest neighbor if a closer one is found
            if distance < min_distance:
                min_distance = distance
                nn_index = midx

        P_M_OMEGA[nn_index] += likelihood(omega, data_list[nn_index], n_pix)
    end_time = time.perf_counter() - start_time
    logger.info("all_pairs_likelihood time: {}".format(end_time))
    return P_M_OMEGA, P_omega_m


def write_csv(single_point_likelihood, area_likelihood, filename="out.csv"):
    f = open(filename, "w")

    head = f"single_point_likelihood, area_likelihood,\n"
    f.write(head)
    for i, sp in enumerate(single_point_likelihood):
        f.write(f"{sp},{area_likelihood[i]},\n")
    f.close()
