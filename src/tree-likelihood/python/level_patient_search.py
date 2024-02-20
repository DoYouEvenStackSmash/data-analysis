#!/usr/bin/python3
from clustering_driver import *
from clustering_imports import *
from tree_search_operations import *
from likelihood_helpers import *


def difference(m1, m2, noise=1):
    return jnp.sqrt(jnp.sum(((m1 - m2) / noise) ** 2))


def diff(m1, m2, noise):
    return float(
        jnp.sqrt(
            jnp.sum(((m1.flatten() - m2.astype(jnp.float32).ravel()) / noise) ** 2)
        ).astype(jnp.float32)
    )


def psearch_leaf(T, idx, data_list, dbest, nearest_neighbor, noise, taow=None,nnq=None):
    """Leaf search function for level patient search

    Args:
        T (DatumT): Target input element
        idx (int): Index of comparison element in data_list
        data_list ([DatumT]): List of data refs in tree
        dbest (float): Best distance seen so far
        nearest_neighbor (int): Index of best neighbor seen so far
        noise (float): some normalization parameter
        TAU (int, optional): a bound for pruning the tree. Defaults to 1.
    """
    res = jax_apply_d1m2_to_d2m1(T, data_list[idx])
    dist = difference(T.m1, res, noise)
    if dist < taow[0]:
        nnq.append((idx,dist))
    if dist < dbest[0]:
        dbest[0] = dist
        nearest_neighbor[0] = idx

def qsearch_leaf(T, idx, data_list, dbest, nearest_neighbor, noise, TAU=1, nnq=None):
    if nnq == None:
        return
    res = jax_apply_d1m2_to_d2m1(T, data_list[idx])
    dist = diff(T.m1, res, noise)
    nnq.append((idx, dist))

    
def level_patient_search(node_list, data_list, input_list, tau=0.4, tauprops=None):
    """Calculates likelihood using a lexicographial breadth first traversal with a bound tau

    Args:
        node_list ([ClusterTreeNode]): List of tree nodes
        data_list ([DatumT]): List of tree data refs
        input_list ([DatumT]): List of inputs potentially with ctfs
        TAU (float, optional): Some parameter for bounding the traversal. Defaults to 0.4.

    Returns:
        _type_: _description_
    """
    noise = calculate_noise(input_list)
    lambda_square = noise**2
    print(noise)
    likelihood_prime = [0.0 for _ in range(len(input_list))]
    likelihood_idx = [0 for _ in range(len(input_list))]
    nq = deque()
    md = 10
    start_time = time.perf_counter()
    sortkey = lambda x: -x[1] if x[1] != 0 else -1 * float("Inf")
    k = len(node_list[0].children)
    tau_fac = 0.05
    # TAU = tau
    
    for i, T in enumerate(input_list):
        TAU = 5
        
        if tauprops != None:
            TAU = difference(T.m1, jax_apply_d1m2_to_d2m1(T, data_list[tauprops[i]]),noise)
            # TAU = node_list[tauprops[i]].cluster_radius
        taow = [TAU]
        # rq
        dbests = []
        nns = []
        skey = lambda x: x[1]
        prev_index = 0
        node_index = 0
        cq = deque()
        sq = deque()
        offt = node_list[0].cluster_radius
        nnq = []
        for c in node_list[0].children:
            cq.append((c, -node_list[c].cluster_radius))
            
            sq.extend(sorted(cq, key=sortkey))
            cq = deque()
            dq = deque()
            dq.append(0)
            # cq.append((None, None))
            depth_counter = 1

            nn = [None]
            mdist = [float("Inf")]
            dbest = [float("Inf")]
            # dbest[0] = TAU
            nn[0] = tauprops[i]
            while len(cq) or len(sq):
                while len(cq):
                    ncounter = 0
                    while cq[0][0] != None:
                        pref_index = node_index
                        node_index, Rprev = cq.popleft()

                        for c in node_list[node_index].children:
                            if node_list[c].data_refs != None:                               
                                
                                for index in node_list[c].data_refs:
                                    psearch_leaf(T, index, data_list, dbest, nn, noise, taow,nnq)
                                    taow[0] = min(taow[0], dbest[0])
                                    
                                continue
                            elif node_list[c].cluster_radius == 0:

                                for index in node_list[c].children[0].data_refs:
                                    psearch_leaf(T, index, data_list, dbest, nn, noise, taow,nnq)
                                    taow[0] = min(taow[0], dbest[0])

                                continue

                            R = node_list[node_index].cluster_radius# / noise
                            res = jax_apply_d1m2_to_d2m1(T, node_list[node_index].val)
                            C = difference(T.m1, res, noise)
                            C2 = difference(node_list[node_index].val.m1, node_list[c].val.m1,noise)

                            if abs(C - C2) <= max(node_list[c].cluster_radius + depth_counter * 0.02, taow[0]):# + R/k:
                                res2 = jax_apply_d1m2_to_d2m1(T, node_list[c].val)
                                C3 = difference(T.m1, res2, noise)
                                if C3 < max(node_list[c].cluster_radius + depth_counter * 0.02, taow[0]):
                                    cq.append((c, -C3))


                    cq.popleft()
                    if len(cq):
                        depth_counter+=1
                        # if dbest[0] == float('inf'):
                        if depth_counter > 2:
                            dq.extend([depth_counter for _ in range(len(cq))])
                            sq.extend(sorted(cq, key=sortkey))#[0:max(k,k**(depth_counter-1))])
                            cq = deque()
                        else:
                            cq = deque(sorted(cq, key=sortkey))
                            depth_counter += 1
                            cq.append((None, None))
                
                while len(sq):
                    depth_counter = dq.pop()
                    # sn_index, Rprev = sq.pop()
                    cq.append(sq.pop())
                    cq.append((None, None))
                    break
                    
                    
            dbests.append((nn[0], dbest[0]))
        sortkey2 = lambda x: x[1]
        dbests = sorted(dbests, key=sortkey2)
        nns.append(dbests[0][0])
        nnq = sorted(nnq, key=sortkey2)
        if len(nnq):
            nns[-1] = nnq[0][0]
        if nns[0] == None:
            continue
        likelihood_prime[i] = jnp.exp(
            -1.0
            * (difference_calculation(T.m1, data_list[nns[0]].m1, noise) ** 2)
            / (2 * lambda_square)
        )
        likelihood_idx[i] = nns[0]
        if not i % 20:
            print(i)
    likelihood_prime = postprocessing_adjust(likelihood_prime, noise, 1)
    end_time = time.perf_counter() - start_time
    LOGGER.info("{},{}".format(len(input_list) ** 2, end_time))
    return likelihood_prime, likelihood_idx
