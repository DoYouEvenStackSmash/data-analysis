#!/usr/bin/python3
from clustering_driver import *
from clustering_imports import *
from tree_search_operations import *
from likelihood_helpers import *


def greedy_tree_likelihood(node_list, data_list, input_list):
    likelihood_prime, likelihood_idx = _greedy_tree_traversal(
        node_list, data_list, input_list
    )
    return likelihood_prime, likelihood_idx

def difference(m1, m2, noise=1):
    return jnp.sqrt(jnp.sum(((m1 - m2) / noise) ** 2))

def _greedy_tree_traversal(node_list, data_list, input_list):
    """
    DFS for the cluster containing(ish) T
    returns an index to a node in node_list
    """
    n_pix = data_list[0].m1.shape[1] ** 2.0

    approx_scale_constant = len(data_list)
    weight = 1  # / len(data_list)
    # Likelihood array
    likelihood_prime = [0.0 for _ in range(len(input_list))]
    likelihood_idx = [0 for _ in range(len(input_list))]
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
                dist = difference(node_list[i].val.m1, T.m1, noise)
                # dist = custom_distance(node_list[i].val, T)
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
            res = jax_apply_d1m2_to_d2m1(T, data_list[idx])
            d = DatumT()
            d.m1 = res
            # dist = custom_distance(d, T)
            dist = difference(T.m1, res, noise)

            if dist < min_dist:
                closest_idx = idx
                min_dist = dist
        # likelihood
        likelihood_prime[input_idx] = weight * jnp.exp(
            -1.0 * (jnp.square(min_dist) / (2 * lambda_square))
        )
        likelihood_idx[input_idx] = closest_idx
    end_time = time.perf_counter() - start_time
    LOGGER.info("{},{}".format(len(input_list), end_time))
    likelihood_prime = postprocessing_adjust(likelihood_prime, noise, 1)
    # likelihood_omega_m = np.array([np.array([a, b]) for a, b in zip(likelihood_prime, likelihood_idx)])
    return likelihood_prime, likelihood_idx
