#!/usr/bin/python3
from clustering_driver import *
from clustering_imports import *
from tree_search_operations import *
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


def postprocessing_adjust(input_arr, noise, const_factor=0):
    """
    Placeholder for modifying an array
    -\log \left(
        \sqrt{2\pi \lambda^2}
    \right)
    """
    log_add = 0
    const_factor = 1

    # branching for easy enable/disable
    if True:
        log_add = np.log(const_factor * np.sqrt(2 * np.pi * noise**2))

    return [np.log(i) - log_add for i in input_arr]


def likelihood(omega, m, N_pix, noise=1):
    """
    New likelihood
    \log \left(
        \sum_{j=1}^M \weight_j
        \exp \left( - \left\| \omega_i - x_j \right\|^2 / 2 \lambda^2 \right)
    \right)
    """
    lambda_square = noise**2
    coeff = 1
    l2 = jnp.exp(-1.0 * (jnp.square(custom_distance(omega, m)) / (2 * lambda_square)))
    return coeff * l2


def search_level_tree(node_list, data_list, T):
    nearest_level_clusters = level_order_search(node_list, T)
    return nearest_level_clusters
    totsum = 0
    for level in nearest_level_clusters:
        minsum = 0
        for cluster_id in level:
            nn = 0
            min_dist = float("inf")
            for i in node_list[cluster_id].data_refs:
                dist = custom_distance(node_list[i].val, T)
                if dist < min_dist:
                    nn = i
                    min_dist = dist
            minsum += min_dist
        totsum += minsum
    return totsum


def evaluate_tree_level_likelihood(node_list, data_list, input_list, noise=1):
    n_pix = data_list[0].m1.shape[1] ** 2.0

    approx_scale_constant = len(data_list)
    weight = 1  # / len(data_list)
    # Likelihood array
    likelihood_omega_m = [0.0 for _ in range(len(input_list))]

    start_time = time.perf_counter()

    for input_index, omega in enumerate(input_list):
        nlc = search_level_tree(node_list, data_list, omega)
        print(nlc)
        for level in nlc:
            for cn_idx in level:
                cluster_node = node_list[cn_idx]
                min_dist = float("Inf")
                om_idx = -1
                for data_idx in cluster_node.data_refs:
                    if custom_distance(data_list[data_idx], omega) < min_dist:
                        min_dist = custom_distance(data_list[data_idx], omega)
                        om_idx = data_idx

                likelihood_omega_m[input_index] += weight * likelihood(
                    omega, data_list[om_idx], n_pix, noise
                )
    likelihood_omega_m = postprocessing_adjust(likelihood_omega_m, noise, 1)

    end_time = time.perf_counter() - start_time
    logger.info("{},{}".format(len(input_list), end_time))

    return likelihood_omega_m


def evaluate_tree_neighbor_likelihood(node_list, data_list, input_list, noise=1):
    """
    Given a hierarchical clustering of the images_from_structures, and the input images,
    evaluate the likelihood of each image with its nearest structure

    Nearest neighbor search
    returns a list of likelihoods associated with the data_list members
    """
    # in application, this is 16384 <- 128 x 128
    n_pix = data_list[0].m1.shape[1] ** 2.0

    approx_scale_constant = len(data_list)
    weight = 1  # / len(data_list)
    # Likelihood array
    likelihood_omega_m = [0.0 for _ in range(len(input_list))]

    start_time = time.perf_counter()

    # omega := experimental image
    for input_index, omega in enumerate(input_list):
        nearest_index, nearest_distance = search_tree(node_list, data_list, omega)
        likelihood_omega_m[input_index] += weight * likelihood(
            omega, data_list[nearest_index], n_pix, noise
        )

    end_time = time.perf_counter() - start_time
    logger.info("{},{}".format(len(input_list), end_time))

    likelihood_omega_m = postprocessing_adjust(likelihood_omega_m, noise, 1)
    return likelihood_omega_m


def evaluate_tree_cluster_likelihood(node_list, data_list, input_list, noise=1):
    """
    Given a hierarchical clustering of the images_from_structures, and the input images,
    Accumulate the likelihoods calculated between each image and the cluster of structures it is associated with

    returns a list of likelihoods associated with the data_list members
    """
    n_pix = data_list[0].m1.shape[1] ** 2

    approx_scale_constant = len(data_list)

    weight = 1  # / len(data_list)

    likelihood_omega_m = [0.0 for _ in range(len(input_list))]  # Likelihood array

    start_time = time.perf_counter()

    # omega := experimental image
    # for each omega, find the cluster it belongs to and calculate the likelihoods of it with all members of the cluster
    for input_index, omega in enumerate(input_list):
        cluster_node = node_list[find_cluster(node_list, omega)]
        for data_idx in cluster_node.data_refs:
            likelihood_omega_m[input_index] += weight * likelihood(
                omega, data_list[data_idx], n_pix, noise
            )

    end_time = time.perf_counter() - start_time
    logger.info("tree_cluster_likelihood time: {}".format(end_time))

    likelihood_omega_m = postprocessing_adjust(likelihood_omega_m, noise, 1)
    return likelihood_omega_m


def calculate_noise(input_list):
    """
    Calculate the noise as the standard deviation
    """
    input_arr = np.array([input_list[i].m1 for i in range(len(input_list))])
    avg = jnp.mean(input_arr)
    noise = jnp.sqrt(
        jnp.divide(
            jnp.sum(jnp.array([jnp.square(x - avg) for x in input_arr])),
            input_arr.shape[0] - 1,
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
    cluster_likelihoods = evaluate_tree_level_likelihood(
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
    n_pix = data_list[0].m1.shape[0] * data_list[0].m1.shape[1]
    likelihood_omega_m = [0.0 for _ in range(len(input_list))]
    approx_scale_constant = len(data_list)
    weight = 1 / len(data_list)
    start_time = time.perf_counter()

    for idx, omega in enumerate(input_list):
        min_distance = float("inf")
        nn_index = 0
        for midx, m in enumerate(data_list):
            distance = custom_distance(omega, m)
            # Update nearest neighbor if a closer one is found
            if distance < min_distance:
                min_distance = distance
                nn_index = midx
        # for i in range(n_pix):
        likelihood_omega_m[idx] += weight * likelihood(omega, m, n_pix, noise)

    end_time = time.perf_counter() - start_time
    logger.info("global_neighbor_likelihood time: {}".format(end_time))

    likelihood_omega_m = postprocessing_adjust(
        likelihood_omega_m, noise, approx_scale_constant
    )
    return likelihood_omega_m


def testbench_likelihood(node_list, data_list, input_list, input_noise=None):
    """
    Testbench function for head to head comparison of tree approximation and global search
    """

    nn_likelihoods = greedy_tree_likelihood(
        node_list, data_list, input_list
    )
    global_likelihoods = bounded_tree_likelihood(
        node_list, data_list, input_list
    )

    return nn_likelihoods, global_likelihoods


def evaluate_global_likelihood(data_list, input_list, noise=1):
    """
    Given reference data and some input data,
    Accumulate the likelihoods according to the paper in the structure indices
    """
    n_pix = data_list[0].m1.shape[1] ** 2.0
    likelihood_omega_m = [0.0 for _ in range(len(input_list))]
    approx_scale_constant = len(data_list)
    weight = 1 / len(data_list)

    start_time = time.perf_counter()

    for idx, omega in enumerate(input_list):
        for midx, m in enumerate(data_list):
            likelihood_omega_m[idx] += weight * likelihood(omega, m, n_pix, noise)

    end_time = time.perf_counter() - start_time

    logger.info("{},{}".format(len(input_list), end_time))
    likelihood_omega_m = postprocessing_adjust(
        likelihood_omega_m, noise, approx_scale_constant
    )
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

    head = f"single_point_likelihood,area_likelihood,\n"
    f.write(head)
    for i, sp in enumerate(single_point_likelihood):
        f.write(f"{sp},{area_likelihood[i]},\n")
    f.close()


def bounded_tree_likelihood(node_list, data_list, input_list, TAU=4e-1):
    n_pix = data_list[0].m1.shape[1] ** 2.0

    approx_scale_constant = len(data_list)
    weight = 1  # / len(data_list)
    # Likelihood array
    likelihood_omega_m = [0.0 for _ in range(len(input_list))]
    noise = calculate_noise(input_list)
    lambda_square = noise**2
    start_time = time.perf_counter()

    LEVEL_FLAG = -1
    q = []
    for input_idx, T in enumerate(input_list):
        q.append(0)
        reachable_cluster_refs = []
        # tracks the level in the tree
        level_counter = 0
        while len(q):
            q.append(LEVEL_FLAG)
            level_counter += 1
            reachable_cluster_refs.append([])

            while q[0] != LEVEL_FLAG:
                elem_id = q.pop(0)
                elem = node_list[elem_id]

                if elem.data_refs != None:
                    reachable_cluster_refs[-1].append(elem_id)
                    continue

                for cidx in elem.children:
                    res_elem_tensor = jax_apply_d1m2_to_d2m1(T, node_list[cidx].val)
                    dist = jnp.linalg.norm(T.m1 - res_elem_tensor)
                    if dist > TAU:
                        continue
                    q.append(cidx)
            # track level end for metrics purposes
            flag = q.pop(0)
        for level in reachable_cluster_refs:
            if not len(level):
                continue
            for cluster_node_idx in level:
                min_dist = float("Inf")
                cluster_node = node_list[cluster_node_idx]
                for idx in cluster_node.data_refs:
                    res = jax_apply_d1m2_to_d2m1(data_list[idx], T)
                    d = DatumT()
                    d.m1 = res
                    dist = custom_distance(d, T)
                    if dist < min_dist:
                        min_dist = dist
                likelihood_omega_m[input_idx] = weight * jnp.exp(
                    -1.0 * (jnp.square(min_dist) / (2 * lambda_square))
                )
    end_time = time.perf_counter() - start_time
    logger.info("{},{}".format(len(input_list), end_time))
    likelihood_omega_m = postprocessing_adjust(likelihood_omega_m, noise, 1)
    return likelihood_omega_m


def greedy_tree_likelihood(node_list, data_list, input_list):
    """
    DFS for the cluster containing(ish) T
    returns an index to a node in node_list
    """
    n_pix = data_list[0].m1.shape[1] ** 2.0

    approx_scale_constant = len(data_list)
    weight = 1  # / len(data_list)
    # Likelihood array
    likelihood_omega_m = [0.0 for _ in range(len(input_list))]
    noise = calculate_noise(input_list)
    lambda_square = noise**2
    coeff = 1

    start_time = time.perf_counter()

    for input_idx, T in enumerate(input_list):
        n_curr = 0
        # search_list = []
        # search representative nodes
        while node_list[n_curr].data_refs is None:
            min_dist = float("inf")
            nn = 0
            for i in node_list[n_curr].children:
                dist = custom_distance(node_list[i].val, T)
                if dist < min_dist:
                    nn = i
                    min_dist = dist
            # search_list.append(nn)
            n_curr = nn
            # search leaves
        closest_idx = 0
        min_dist = float("inf")
        d = None
        for idx in node_list[n_curr].data_refs:
            # print(idx)
            res = jax_apply_d1m2_to_d2m1(data_list[idx], T)
            d = DatumT()
            d.m1 = res
            dist = custom_distance(d, T)
            if dist < min_dist:
                closest_idx = idx
                min_dist = dist
        # likelihood
        likelihood_omega_m[input_idx] = weight * jnp.exp(
            -1.0 * (jnp.square(min_dist) / (2 * lambda_square))
        )
    end_time = time.perf_counter() - start_time
    logger.info("{},{}".format(len(input_list), end_time))
    likelihood_omega_m = postprocessing_adjust(likelihood_omega_m, noise, 1)
    return likelihood_omega_m


def bounded_tree_likelihood(node_list, data_list, input_list, TAU=4e-1):
    n_pix = data_list[0].m1.shape[1] ** 2.0

    approx_scale_constant = len(data_list)
    weight = 1 / len(data_list)
    # Likelihood array
    likelihood_omega_m = [0.0 for _ in range(len(input_list))]
    noise = calculate_noise(input_list)
    lambda_square = noise**2
    start_time = time.perf_counter()

    LEVEL_FLAG = -1
    q = []
    for input_idx, T in enumerate(input_list):
        q.append(0)
        reachable_cluster_refs = []
        # tracks the level in the tree
        level_counter = 0
        while len(q):
            q.append(LEVEL_FLAG)
            level_counter += 1
            reachable_cluster_refs.append([])

            while q[0] != LEVEL_FLAG:
                elem_id = q.pop(0)
                elem = node_list[elem_id]

                if elem.data_refs != None:
                    reachable_cluster_refs[-1].append(elem_id)
                    continue

                for cidx in elem.children:
                    res_elem_tensor = jax_apply_d1m2_to_d2m1(T, node_list[cidx].val)
                    dist = jnp.linalg.norm(T.m1 - res_elem_tensor)
                    if dist > TAU:
                        continue
                    q.append(cidx)
            # track level end for metrics purposes
            flag = q.pop(0)
        for level in reachable_cluster_refs:
            if not len(level):
                continue
            for cluster_node_idx in level:
                min_dist = float("Inf")
                cluster_node = node_list[cluster_node_idx]
                for idx in cluster_node.data_refs:
                    res = jax_apply_d1m2_to_d2m1(data_list[idx], T)
                    d = DatumT()
                    d.m1 = res
                    dist = custom_distance(d, T)
                    if dist < min_dist:
                        min_dist = dist
                    likelihood_omega_m[input_idx] = weight * jnp.exp(
                        -1.0 * (jnp.square(dist) / (2 * lambda_square))
                    )
    end_time = time.perf_counter() - start_time
    logger.info("{},{}".format(len(input_list), end_time))
    likelihood_omega_m = postprocessing_adjust(likelihood_omega_m, noise, 1)
    return likelihood_omega_m
