#!/usr/bin/python3
from clustering_driver import *
from clustering_imports import *
from tree_search_operations import *
from likelihood_helpers import *


def diff(m1, m2, noise):
    return float(
        jnp.sqrt(
            jnp.sum(((m1.flatten() - m2.astype(jnp.float32).ravel()) / noise) ** 2)
        ).astype(jnp.float32)
    )


def new_search_leaf(
    T, idx, data_list, dbest, nearest_neighbor, noise, parent, mdist, TAU=1
):
    dist = diff(T.m1, data_list[idx].m1, noise)
    if dist < dbest[0]:
        # parent[0] = parent
        dbest[0] = dist
        mdist[0] = diff(parent.val.m1, T.m1, noise)
        nearest_neighbor[0] = idx


def new_search_node(
    T, node_index, node_list, data_list, irad, dbest, nn, noise, TAU, depth, mdist
):
    if node_list[node_index].data_refs != None:
        for index in node_list[node_index].data_refs:
            new_search_leaf(
                T, index, data_list, dbest, nn, noise, node_list[node_index], mdist, TAU
            )
        return
    if node_list[node_index].data_refs == None and irad < mdist[0]:
        BL = []
        for c in node_list[node_index].children:
            # if node_list[c].cluster_radius ==0:
            #   BL.append((c,0))
            #   continue

            R = node_list[c].cluster_radius / noise
            C = diff(T.m1, node_list[c].val.m1, noise)
            # nc = node_list[node_index].cluster_radius

            # if new_bound  R and R !=0:
            # continue

            # new_bound = 0
            # # if C > R + irad:
            # #   continue
            # if C > irad + R and R != 0:
            #   new_bound = (R - C)*noise

            # # if C < Rprev:
            # #   continue
            # # if C > Rprev:
            # #   continue
            # elif C > R and R != 0:
            #   new_bound = min(irad,2*R)*noise
            #   # new_bound = min(C-Rprev - R), 2 * R)*noise
            # elif C < R:
            #   new_bound = (R - C) * noise
            # if new_bound != 0:
            BL.append((c, (R - C) * noise))
        # BL = [(c,noise*(node_list[c].cluster_radius/noise - diff(T.m1, node_list[c].val.m1, noise))) for c in node_list[node_index].children]
        # if axis == 0:
        sortkey = lambda x: -x[1]
        # if axis == 1:
        #   sortkey = lambda x: x[1]
        dbl = sorted(BL, key=sortkey)
        # reversed(dbl)
        dbl = [dbl[c] for c in range(len(dbl))]
        # sidx = [i for i in range(1,len(BL))]
        if not len(dbl):
            return

        new_search_node(
            T,
            dbl[0][0],
            node_list,
            data_list,
            dbl[0][1],
            dbest,
            nn,
            noise,
            TAU,
            depth + 1,
            mdist,
        )

        z = 1
        while z < min(len(dbl), 3):
            new_search_node(
                T,
                dbl[z][0],
                node_list,
                data_list,
                dbl[z][1],
                dbest,
                nn,
                noise,
                TAU,
                depth + 1,
                mdist,
            )
            z += 1


def impatient_tree_likelihood(node_list, data_list, input_list):
    noise = calculate_noise(input_list)
    lambda_square = noise**2
    likelihood_prime = [0.0 for _ in range(len(input_list))]
    likelihood_idx = [0 for _ in range(len(input_list))]
    sortkey = lambda x: x[0]
    start_time = time.perf_counter()
    for i, T in enumerate(input_list):
        nnl = []
        dbests = []
        metrics = {"change_count": 0}
        for c in reversed(node_list[0].children):
            nn = [None]
            mdist = [float("Inf")]
            dbest = [float("Inf")]
            init_d = node_list[0].cluster_radius / noise
            mdist = [float("inf")]
            new_search_node(
                T, c, node_list, data_list, init_d, dbest, nn, noise, 1, 0, mdist
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
        if not i % 10:
            print(i)
        # print(i)
    likelihood_prime = postprocessing_adjust(likelihood_prime, noise, 1)
    end_time = time.perf_counter() - start_time
    LOGGER.info("{},{}".format(len(input_list) ** 2, end_time))

    return likelihood_prime, likelihood_idx
