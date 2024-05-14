# from sklearn.cluster import KMeans
import numpy as np
import torch
from collections import deque
from fast_pytorch_kmeans import KMeans

from aux_functions import *
from debugging import *
from torch_traversal import *

def get_data_terms(data, k=2,max_t = 15):
    # initialize kvalues
    kterms = np.arange(max_t)
    terms = []
    i = 0
    a = 1
    # populate terms with geometric series
    while i == 0 or terms[-1] <= len(data):
        terms.extend([np.sum(a * k ** kterms[0 : i + 1])])
        i += 1
    # add one extra to prevent corruption
    terms.extend([np.sum(a * k ** kterms[0 : i + 1])])
    return terms

def scatter_tree_build(data, k=2):
    # TODO: preprocess data
    # data = torch.tensor([i.ravel() for i in data])
    # print(len(data))
    o_data_indices = np.arange(data.shape[0])

    local_block_q = deque([i for i in range(1)])
    local_block_q.append(None)

    base_addr_q = deque([0 for _ in range(1)])
    base_addr_q.append(None)

    data_idx_q = deque([o_data_indices])
    data_idx_q.append(None)

    kt = get_data_terms(data, k)
    print(kt)
    print(k * kt[-1])
    row_lens = [(kt[i + 1] - kt[i]) for i in range(len(kt) - 1)]

    global_data = [None] * (len(data) + (kt[-1]))
    tree_idx = [None] * (len(data) + (kt[-1]))
    radius_data = [None] * (len(data) + kt[-1])
    global_data[0 : 0 + len(data)] = data

    base_offt = len(data)
    counter = 0
    while len(local_block_q):
        while local_block_q[0] != None:
            # pop elements from queues
            data_indices = data_idx_q.popleft()
            parent_idx = local_block_q.popleft()
            curr_level = base_addr_q.popleft()

            # compute start of tree level in the array
            base_addr = base_offt + sum(row_lens[0:curr_level])
            # compute position of parent's child block in the tree level
            block_posn = (parent_idx) * k

            # compute position of target block in the array
            wstart = base_addr + block_posn
            # compute the end of target block in the array. should aways be aligned with k
            wend = min(wstart + k, len(global_data))

            # # # short circuit if one of the first data_indices is less than k
            # if len(data_indices) <= k:
            #     # assign leaf indices as negative numbers to indicate special status
            #     for i in range(len(data_indices)):
            #         tree_idx[wstart + i] = -1 * (data_indices[i] + 1)
            #     continue
            
            # perform kmeans
            kmeans = KMeans(n_clusters=k,max_iter =300)
            labels = kmeans.fit_predict(data[data_indices])

            # assign cluster centers and their indices to global_data and
            # tree_idx respectively
            sm = {}
            last = -1
            hold = 0
            idx = []
            for i, c in enumerate(kmeans.centroids):
                if torch.all(torch.eq(c, 0)):
                    continue
                sm[i] = len(idx)
                idx.append(i)
                global_data[wstart + sm[i]] = c
                tree_idx[wstart + sm[i]] = sm[i]
            # collect cluster members
            associated_data = [deque() for _ in range(len(kmeans.centroids))]
            for i, elem in enumerate(labels):
                associated_data[elem].append(data_indices[i])
            
            # calculate radii
            for i,c in enumerate(kmeans.centroids):
                if torch.all(torch.eq(c, 0)):
                    continue
                # print(associated_data[i])
                radius_data[wstart+sm[i]] = torch.max(calculate_radius(c, data[torch.tensor(associated_data[i])],0))
            # handle each cluster
            for i,elem in enumerate(associated_data):
                if len(elem) == 0:
                    continue
                # special case for when the cluster is less than k
                if len(elem) == 1:
                    tree_idx[wstart + sm[i]] = -1 * (elem[0] + 1)
                    global_data[wstart + sm[i]] = None
                    continue
                if len(elem) <= k and len(elem) > 0:
                    # same as above except we are looking down into the next row
                    _base_addr = sum(row_lens[0 : curr_level + 1]) + base_offt
                    _block_posn = (block_posn + sm[i]) * k
                    _wstart = _base_addr + _block_posn
                    # we may need to realloc.
                    # TODO: this can introduce bad behaviors, should be looked at carefully
                    if _wstart + len(elem) >= len(tree_idx):
                        tree_idx.extend([None] * (_wstart + len(elem) - len(tree_idx)))
                        global_data.extend([None] * (len(tree_idx) - len(global_data)))
                        radius_data.extend([None] * (len(tree_idx) - len(radius_data)))
                    # assign leaf indices as negative numbers to indicate special status
                    for j, e in enumerate(elem):
                        tree_idx[_wstart + j] = -1 * (elem[j] + 1)
                    continue
                else:
                    # push the necessary things onto queues
                    local_block_q.extend([block_posn + sm[i]])
                    base_addr_q.extend([curr_level + 1])
                    data_idx_q.extend([associated_data[i]])

        # handle end of level
        local_block_q.popleft()
        base_addr_q.popleft()
        data_idx_q.popleft()
        if len(local_block_q):
            local_block_q.append(None)
            base_addr_q.append(None)
            data_idx_q.append(None)

    return tree_idx, global_data,radius_data, kt, row_lens

def gather_tree_dictionary(tree_idx, global_data, base_offt, k, kt, row_lens):
    local_block_q = deque([i for i in range(1)])
    local_block_q.append(None)
    base_addr_q = deque([0 for _ in range(1)])
    base_addr_q.append(None)
    depth = 1
    tree_dict = {}
    while len(local_block_q):
        while local_block_q[0] != None:
            parent_idx = local_block_q.popleft()
            curr_level = base_addr_q.popleft()

            # compute start of tree level in the array
            base_addr = sum(row_lens[0:curr_level]) + base_offt
            # compute position of parent's child block in the tree level
            block_posn = (parent_idx) * k
            # compute position of target block in the array
            wstart = base_addr + block_posn
            # compute the end of target block in the array. should always be aligned with k
            wend = wstart + k

            # traverse block
            for curr_idx, e in enumerate(tree_idx[wstart:wend]):
                child_indices = []
                if e == None:  # empty node
                    continue
                if e < 0:  # leaf node
                    # report the correct index by inverting the negative offset
                    child_indices.append(-1 * (e + 1))

                    # checksum
                    tree_idx[-1 * (e + 1)] = (
                        -1
                        if tree_idx[-1 * (e + 1)] == None
                        else tree_idx[-1 * (e + 1)] - 1
                    )
                else:  # internal node
                    # check next row addresses to see if they are valid children
                    for child_idx in range(k):
                        # TODO: list[A:B+1] allows B to be greater than list length
                        # could be the source of bugs for k=2
                        # compute array address of child node(same as above)
                        child_addr = (
                            sum(row_lens[0 : curr_level + 1])
                            + base_offt
                            + (block_posn + curr_idx) * k
                            + child_idx
                        )
                        if child_addr >= len(tree_idx) or tree_idx[child_addr] != None:
                            child_indices.append(child_addr)
                    # add necessary things to queues
                    local_block_q.extend([block_posn + curr_idx])
                    base_addr_q.extend([curr_level + 1])

                # add child information to tree dictionary
                if len(child_indices) and wstart + curr_idx not in tree_dict:
                    tree_dict[(wstart + curr_idx)] = []
                    tree_dict[(wstart + curr_idx)].extend(child_indices)

        # handle end of level
        local_block_q.popleft()
        base_addr_q.popleft()
        if len(local_block_q):
            local_block_q.append(None)
            base_addr_q.append(None)
    return tree_dict

def tree_minimize(data, tree_dict, tree_idx, global_data, radius_data, N, k):
    new_tree_dict = {}
    new_global_data = []
    new_radius_data = []
    node_lookup = {}
    new_global_data.extend(data)
    new_radius_data.extend([None]*len(data))
    # print(list(tree_dict))
    node_blacklist = set()
    for k in list(tree_dict):
        # print(k)
        if k not in node_lookup:
            if len(tree_dict[k]) == 1 and tree_dict[k][0] < N:
                node_lookup[k] = tree_dict[k][0]
                node_blacklist.add(k)
                continue
            node_lookup[k] = len(new_global_data)
            new_global_data.append(global_data[k])
            new_tree_dict[node_lookup[k]] = []
            new_radius_data.append(radius_data[k])

    for k,v in tree_dict.items():
        if k in node_blacklist:
            continue
        for i,val in enumerate(v):
            if val < N:
                new_tree_dict[node_lookup[k]].append(val)
            else:
                if val not in node_lookup:
                    continue
                new_tree_dict[node_lookup[k]].append(node_lookup[val])
    return new_tree_dict,new_global_data,new_radius_data
            