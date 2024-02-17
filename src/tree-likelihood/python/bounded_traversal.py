#!/usr/bin/python3
from clustering_driver import *
from clustering_imports import *
from tree_search_operations import *
from likelihood_helpers import *

import scipy.io


def bounded_tree_likelihood(node_list, data_list, input_list):
    print(len(input_list))
    print(len(node_list))
    # (input_list, data_list, 2) where the 2 are (likelihood, index)
    likelihood_matrix, likelihood_idx = _unbounded_tree_traversal(
        node_list, data_list, input_list
    )
    scipy.io.savemat(
        "bounded_likelihood_data.mat",
        {
            "bounded_likelihood_matrix": likelihood_matrix,
            "bounded_index_mat": likelihood_idx,
        },
    )
    # bounded_likelihood_mat = [
    #     a[0][0] for a in _unbounded_tree_likelihood(node_list, data_list, input_list)
    # ]
    # return bounded_likelihood_mat
    # compare_tree_likelihoods(node_list, data_list, input_list)
    # sys.exit()


def _unbounded_tree_traversal(node_list, data_list, input_list, TAU=0.4):
    start_time = time.perf_counter()
    likelihood_omega_m = [
        None for _ in range(len(input_list))
    ]  # initialize likelihood for each input image
    likelihood_idx = [None for _ in range(len(input_list))]
    noise = calculate_noise(input_list)  # calculate noise
    lambda_square = noise**2
    for input_index, T in enumerate(input_list):
        reachable_idx = []
        reachable_structures = []  # initialize some things
        q = deque()  # initialize some things
        q.append(0)  # insert root index

        while len(q):  # get all reachable structures
            elem_id = q.popleft()
            elem = node_list[elem_id]
            if elem.data_refs != None:  # isLeaf = True
                reachable_structures.extend(  # insert children as reachable structures
                    [data_list[idx].m1 for idx in elem.data_refs]
                )
                reachable_idx.extend([idx for idx in elem.data_refs])
                continue
            else:  # isLeaf = False
                q.extend(elem.children)  # add children back into traversal queue
        print(input_index, len(reachable_structures))

        # at this point we have all the reachable structures(which should be all of them)
        # we then calculate the likelihood of each
        # difference calculation:
        likelihood_prime = [  # calculate likelihood of each structure in logspace
            jnp.exp(
                -1
                * (
                    (
                        difference_calculation(T.m1, rs, noise) ** 2
                    )  # jnp.sqrt(jnp.sum(((m1 - m2) / noise) ** 2))
                    / (2 * lambda_square)
                )
            )
            for rs in reachable_structures
        ]
        # selector = lambda val: val if val < TAU else 0

        # likelihood_prime = [selector(rs) for rs in likelihood_prime]

        # we then add the offset factor because we're in logspace
        likelihood_prime = postprocessing_adjust(
            likelihood_prime, noise, 1
        )  # apply postprocessing factor

        # (input_list, data_list, 2) where the 2 are (likelihood, index)

        # likelihood_omega_m[input_index] = np.array(
        #     [
        #         np.array([like, idx])
        #         for like, idx in zip(likelihood_prime, reachable_idx)
        #     ]
        # )
        likelihood_omega_m[input_index] = np.real(likelihood_prime)
        likelihood_idx[input_index] = np.array(reachable_idx)
        # for this contrived example it is assumed to be all
    end_time = time.perf_counter() - start_time

    LOGGER.info("{},{}".format(len(input_list) ** 2, end_time))
    return np.array(likelihood_omega_m), np.array(likelihood_idx)
