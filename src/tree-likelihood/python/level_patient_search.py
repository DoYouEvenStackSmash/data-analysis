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
    dist = difference(T.m1, res, 1)
    # if dist < taow[0]:
    #     nnq.append((idx,dist))
    if dist < dbest[0]:
        dbest[0] = dist
        nearest_neighbor[0] = idx

def qsearch_leaf(T, idx, data_list, dbest, nearest_neighbor, noise, TAU=1, nnq=None):
    if nnq == None:
        return
    res = jax_apply_d1m2_to_d2m1(T, data_list[idx])
    dist = diff(T.m1, res, noise)
    nnq.append((idx, dist))

def check_ctf_bound(omega, mulk, yi,noise=1,tau=1e-6):
    # R = centroid.cluster_radius
    # -2 * noise*log(tau)
    # print(difference(omega.val.m1, yi.m1) - difference(yi.m1, jax_apply_d1m2_to_d2m1(yi,mulk.val)))
    if difference(omega.val.m1, yi.m1) > difference(yi.m1, jax_apply_d1m2_to_d2m1(yi,mulk.val)) - mulk.cluster_radius:
        return True
    return False
    #(||yi - center|| - bound) ** 2 >=  -2 * noise*log(tau)
    # Cm = jax_apply_d1m2_to_d2m1(centroid.val, centroid.val)
    # max_filter_yi = jax_apply_d1m2_to_d2m1(centroid.val, T)
    # ||F(yi) - F(omega)|| >= ||F(yi) - Cphi*F(mulk)|| - max||Cmax*F(omega_j_in_cluster) - Cmax*F(mulk)||
    # || yi - xi|| >= ||yi - Cp * mulk|| - max_over_j||Cmax*F(omega_j_in_cluster) - Cmax*mulk||
    # || yi - xi|| >= difference(T.m1, jax_apply_d1m2_to_d2m1(T,mulk.val)) - mulk.cluster_radius
  
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
        TAU = 85
        
        if tauprops != None:
            if tauprops[i] > 0:
                    
                TAU = difference(T.m1, jax_apply_d1m2_to_d2m1(T, data_list[tauprops[i]]),noise).real
                # print(TAU)
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
            FIRST = True
            nn = [None]
            mdist = [float("Inf")]
            dbest = [float("Inf")]
            # dbest[0] = TAU
            # nn[0] = tauprops[i]
            FIRED = False
            while len(cq) or len(sq):
                while len(cq):
                    ncounter = 0
                    while cq[0][0] != None:
                        pref_index = node_index
                        node_index, Rprev = cq.popleft()
                        if node_list[node_index].children == None:
                            print(node_list[node_index].data_refs)
                        for c in node_list[node_index].children:
                            # if data_refs is none, this is a leaf node
                            
                            if node_list[c].data_refs != None:                               
                                for index in node_list[c].data_refs:
                                    psearch_leaf(T, index, data_list, dbest, nn, noise, taow,nnq)
                                    taow[0] = min(taow[0], dbest[0])
                                if FIRST:
                                    # reversed(sq)
                                    FIRST = False
                                    # break
                                continue
                            # An additional check for simple paths.
                            # If cluster radius is zero, this is a leaf node. 
                            elif node_list[c].cluster_radius == 0:
                                
                                for index in node_list[c].children[0].data_refs:
                                    psearch_leaf(T, index, data_list, dbest, nn, noise, taow,nnq)
                                    taow[0] = min(taow[0], dbest[0])
                                if FIRST:
                                    # reversed(sq)
                                    FIRST = False
                                    # break
                                continue                  
                        
                            # geometry things
                            if check_ctf_bound(node_list[c],node_list[node_index], T, noise):
                                cq.append((c,1))                                  
                            else:
                                # print("fire")
                                pass

                    
                    if not FIRST and not FIRED:
                        FIRED = True
                        while cq[0][0] != None:
                            print("fir2e")
                            sq,append(cq.popleft())
                            dq.append(depth_counter)
                        # reversed(sq)
                        # reversed(dq)
                        # break
                    # pop the none off of the queue to signify end of level
                    cq.popleft()
                    if len(cq):
                        
                        # depth_counter
                        if depth_counter > 2: # if depth exceeds 2, start doing lexBFS
                            val = len(sq)
                            # print(cq)
                            sq.extend(sorted(cq, key=sortkey)[0:max(k,int(len(cq)/3))])
                            dq.extend([depth_counter for _ in range(len(sq) - val)])
                            cq = deque()
                        else:
                            cq = deque(sorted(cq, key=sortkey))
                            depth_counter += 1
                            cq.append((None, None))
                
                # lexicographical bfs
                while len(sq):
                    # print(sq)
                    # print(dq)
                    if sq[0][0] == None:
                        # print("fire")
                        dq.pop()
                        cq.append(sq.pop())
                        cq.append((None, None))

                        break
                        
                    # print(len(sq))

                    # print(depth_counter)
                    counter = 0
                    if len(dq) > 1:
                        # print(depth_counter)
                        while len(dq) > 1 and depth_counter == dq[-1]:
                            depth_counter = dq.pop()
                            cq.append(sq.pop())
                            counter +=1
                    else:
                        depth_counter = dq.pop()
                        cq.append(sq.pop())

                    
                    # if not len(dq) and len(sq)
                    # sn_index, Rprev = sq.pop()
                    # cq.append(sq.pop())
                    # if counter > 1:
                    #     print(counter)
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
            * (difference_calculation(T.m1, jax_apply_d1m2_to_d2m1(T, data_list[nns[0]]), noise) ** 2)
            / (2 * lambda_square)
        )
        likelihood_idx[i] = nns[0]
        if not i % 20:
            print(i)
    likelihood_prime = postprocessing_adjust(likelihood_prime, noise, 1)
    end_time = time.perf_counter() - start_time
    LOGGER.info("{},{}".format(len(input_list) ** 2, end_time))
    return likelihood_prime, likelihood_idx
