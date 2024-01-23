def bounded_dist_bfs(node_list, T, TAU=4e-1):
    """Level order traversal of the hierarchical clustering for valid clusters for T within
    distance TAU

    Args:
        node_list (list of ClusterTreeNodes): _description_
        T (DatumT): _description_
        TAU (_t, optional): _description_. Defaults to 1e-8.

    Returns:
        reachable_cluster_refs (list of ints): The reference indices of the clusters within TAU
    """
    LEVEL_FLAG = -1
    q = []
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

    return reachable_cluster_refs


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
    weight = 1  # / len(data_list)
    # Likelihood array
    likelihood_omega_m = [0.0 for _ in range(len(input_list))]
    noise = calculate_noise(input_list)
    lambda_square = noise**2

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
