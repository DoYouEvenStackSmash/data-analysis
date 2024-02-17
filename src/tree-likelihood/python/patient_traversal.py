#!/usr/bin/python3
from clustering_driver import *
from clustering_imports import *
from tree_search_operations import *
from likelihood_helpers import *


def difference(m1, m2, noise=1):
    return jnp.sqrt(jnp.sum(((m1 - m2) / noise) ** 2))


searchcount = 0
exclude_count = {}
node_searchcount = 0
neg_search = 0
levels = {}


def search_leaf(
    T, idx, data_list, dbest, nearest_neighbor, noise, nndist, mdist, metrics, TAU=1
):
    global searchcount
    searchcount += 1
    global f
    dist = difference(T.m1, data_list[idx].m1, noise)
    if dist < dbest[0]:
        dbest[0] = dist
        metrics["change_count"] += 1
        nearest_neighbor[0] = idx
        mdist[0] = nndist


def search_node(
    T,
    node_index,
    node_list,
    data_list,
    inherited_radius,
    dbest,
    nearest_neighbor,
    mdist,
    metrics,
    noise=1,
    TAU=1,
    depth=0,
):
    global exclude_count
    global node_searchcount
    node_searchcount += 1

    if (
        node_list[node_index].data_refs
        != None
        # node_list[node_index].cluster_radius == 0
    ):  # isLeaf = True
        for index in node_list[node_index].data_refs:
            search_leaf(
                T,
                index,
                data_list,
                dbest,
                nearest_neighbor,
                noise,
                inherited_radius,
                mdist,
                metrics,
                TAU,
            )
        return
        # return
    if inherited_radius < 0:
        global neg_search
        neg_search += 1
        # if depth not in exclude_count:
        #     exclude_count[depth] = 0
        # exclude_count[depth] += 1
        return
    k = len(node_list[node_index].children)
    if (
        node_list[node_index].data_refs == None and inherited_radius < mdist[0]
    ):  # isLeaf = false and distance to boundary of cluster is less than the best distance so far

        distances_to_cluster_boundary = [  # calculate the next cluster boundary distances
            (
                (
                    (node_list[index].cluster_radius/noise)-
                    difference(T.m1, node_list[index].val.m1, noise)
                ),
                index,
            )
            for i, index in enumerate(node_list[node_index].children)
        ]
        sortkey = lambda x: x[0]
        distances_to_cluster_boundary = sorted(
            distances_to_cluster_boundary, key=sortkey
        )  # sort the cluster boundary distances in decreasing order to account for unbalanced tree
        prime_index = 0
        sub_indices = [
            rdx
            for rdx in range(1, len(distances_to_cluster_boundary))
            if distances_to_cluster_boundary[rdx][0] >= 0
        ]  # collect indices for other children where the boundary distance is less current child distance + sum of all other child distances
        search_node(
            T,
            distances_to_cluster_boundary[prime_index][1],
            node_list,
            data_list,
            distances_to_cluster_boundary[prime_index][0],
            dbest,
            nearest_neighbor,
            mdist,
            metrics,
            noise,
            TAU,
            depth + 1,
        )

        z = 0
        while z < len(sub_indices):
            sub_index = sub_indices[z]
            z += 1
            search_node(
                T,
                distances_to_cluster_boundary[sub_index][1],
                node_list,
                data_list,
                distances_to_cluster_boundary[sub_index][0],
                dbest,
                nearest_neighbor,
                mdist,
                metrics,
                noise,
                TAU,
                depth + 1,
            )
            return
            # sub_index_sum += distances_to_cluster_boundary[sub_index][0]

    if inherited_radius > dbest[0]:
        # pass
        if depth not in exclude_count:
            exclude_count[depth] = 0
        exclude_count[depth] += 1


def patient_tree_likelihood(node_list, data_list, input_list, TAU=0.4):
    likelihood_prime, likelihood_idx = _patient_tree_traversal(
        node_list, data_list, input_list, TAU
    )
    return likelihood_prime, likelihood_idx


def patient_search_tree(node_list, data_list, T, noise, TAU=0.4):
    return
    # noise = 1#calculate_noise([T])
    # lambda_square = noise**2
    nnl = []
    dbests = []
    for c in node_list[0].children:
        dbest = [float("Inf")]
        nn = [None]
        init_d = node_list[c].cluster_radius / noise - np.sqrt(
            (np.sum(T.m1 - node_list[c].val.m1) / noise) ** 2
        )

        search_node(T, c, node_list, data_list, init_d, dbest, nn, noise, TAU, depth=1)
        nnl.append(nn[0])
        dbests.append(dbest[0])
    sortkey = lambda x: x[0]
    md = sorted([(dist, idx) for idx, dist in zip(nnl, dbests)], key=sortkey)
    # md = abc
    min_idx = md[0][1]
    min_dist = np.real(md[0][0])
    return min_idx, min_dist


def _patient_tree_traversal(node_list, data_list, input_list, TAU=0.4):
    nns = []
    noise = calculate_noise(input_list)
    lambda_square = noise**2
    # likelihood_omega_m =
    likelihood_prime = [0.0 for _ in range(len(input_list))]
    likelihood_idx = [0 for _ in range(len(input_list))]
    sortkey = lambda x: x[0]
    start_time = time.perf_counter()
    global searchcount
    global exclude_count
    val = None
    k = len(node_list[0].children)
    for i, T in enumerate(input_list):
        nnl = []
        dbests = []
        metrics = {"change_count": 0}
        for c in reversed(node_list[0].children):
            dbest = [float("Inf")]
            nn = [None]
            mdist = [float("Inf")]
            init_d = k * jnp.log(node_list[0].cluster_radius / noise) - np.sqrt(
                np.sum(((T.m1 - node_list[c].val.m1) / noise) ** 2)
            )

            search_node(
                T,
                c,
                node_list,
                data_list,
                init_d,
                dbest,
                nn,
                mdist,
                metrics,
                noise,
                TAU,
                depth=1,
            )
            nnl.append(nn[0])
            dbests.append(dbest[0])
        val = sorted([(dist, idx) for idx, dist in zip(nnl, dbests)], key=sortkey)[0]

        likelihood_prime[i] = jnp.exp(
            -1.0
            * (difference_calculation(T.m1, data_list[val[1]].m1, noise) ** 2)
            / (2 * lambda_square)
        )
        likelihood_idx[i] = val[1]
        if not i % 20:
            print(neg_search, end="\t")
            print(searchcount, end="\t")
            print(node_searchcount, end="\t")
            print(exclude_count)
        # else:
        #     print(metrics["change_count"])
    likelihood_prime = postprocessing_adjust(likelihood_prime, noise, 1)
    end_time = time.perf_counter() - start_time
    LOGGER.info("{},{}".format(len(input_list) ** 2, end_time))

    return likelihood_prime, likelihood_idx
