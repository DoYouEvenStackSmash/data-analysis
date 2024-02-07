#!/usr/bin/python3
from clustering_driver import *
from clustering_imports import *
from tree_search_operations import *
from likelihood_helpers import *


def bounded_tree_likelihood(node_list, data_list, input_list):
    # (input_list, data_list, 2) where the 2 are (likelihood, index)
    bounded_likelihood_mat = _unbounded_tree_traversal(node_list, data_list, input_list)

    # bounded_likelihood_mat = [
    #     a[0][0] for a in _unbounded_tree_likelihood(node_list, data_list, input_list)
    # ]
    return bounded_likelihood_mat
    # compare_tree_likelihoods(node_list, data_list, input_list)
    # sys.exit()


def _unbounded_tree_traversal(node_list, data_list, input_list, TAU=0.4):
    likelihood_omega_m = [
        [] for _ in range(len(input_list))
    ]  # initialize likelihood for each input image
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

        likelihood_omega_m[input_index] = np.array(
            [
                np.array([like, idx])
                for like, idx in zip(likelihood_prime, reachable_idx)
            ]
        )

        # for this contrived example it is assumed to be all
    return likelihood_omega_m
