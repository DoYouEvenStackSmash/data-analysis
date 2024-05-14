import torch
import numpy as np

from collections import *
from aux_functions import *


mag = lambda x: torch.sqrt(x.real**2 + x.imag**2)

def compact_level_order_search(T,tree_dictionary, radius_data, global_data,ctf, N, k, tau):
    results = deque()
    T = T.cuda()
    ctf = ctf.cuda()

    # initialize node queue with initial indices
    node_q = deque([N + i for i in range(k)])
    node_q.append(None)
    # level counter
    lc = 0
    # perform level order traversal
    while len(node_q):
        # storage array for level values
        # NOTE: NOT storage for images. not immediately.
        level_vals = []
        # print(node_q)
        while node_q[0] != None:
            # # make sure node is actually a real node
            if node_q[0] in tree_dictionary:
                val = node_q.popleft()
                # # if node is a direct path to a leaf, go to the leaf
                # if len(tree_dictionary[val]) == 1 and tree_dictionary[val][0] < N:
                #     v = tree_dictionary[val][0]
                level_vals.extend([val])
            else:
                if node_q[0] < N:
                    level_vals.extend([node_q.popleft()])
                else:
                    # bypass any weird phantom nodes
                    pass
                    # print("???\n\n\n")
                    node_q.popleft()
        node_q.popleft()

        # init storage array for images
        images = []
        # populate image array with actual images(centroids and leaves)
        if len(level_vals):
            # for i,elem in enumerate(level_vals):
            #     if torch.isnan(gd[elem][0]):
            #         level_vals[i] = tree_dictionary[level_vals[i]][0]

            images = torch.tensor(global_data[torch.tensor(level_vals)]).cuda()
            Tp = T.expand_as(images)
            Cp = ctf.expand_as(images)

            distances = calculate_distance(Tp, images,Cp ,0).cpu()

            # distances = mag(calculate_distance(Tp, images,Cp ,0)).cpu()

            # torch.cuda.empty_cache()
        # add new references to node q if they are not phantom nodes
            for i,elem in enumerate(distances):

                if level_vals[i] >= N and elem > radius_data[level_vals[i]]+tau:
                    continue
                if level_vals[i] in tree_dictionary:
                    node_q.extend(tree_dictionary[level_vals[i]])
                if level_vals[i] < N and elem < tau:
                    results.append(level_vals[i])
        # handle end of level
        if len(node_q):
            node_q.append(None)
    return results


def gather_clustered_data(tree_dictionary, global_data, N, k):
    #puts all members of a cluster into the same array
    # initialize global data as numpy array for indexing
    gd = torch.tensor([i for i in global_data], dtype=object)
    # this will hold the clusters
    clusters = []

    # each element in the tree dictionary is (arguably) a cluster
    for elem in list(tree_dictionary):
        point_array = []
        # if elem is a leaf, pass it through
        if elem in tree_dictionary and len(tree_dictionary[elem]) == 1:
            elem = tree_dictionary[elem][0]
        if elem < N: # if elem < N, it's a leaf
            point_array.append(elem)
            clusters.append(gd[torch.tensor(point_array)])
            continue
        # normal level order traversal
        node_q = deque([elem, None])
        while len(node_q):
            while node_q[0] != None:
                if node_q[0] in tree_dictionary:
                    v = node_q.popleft()
                    if len(tree_dictionary[v]) == 1 and tree_dictionary[v][0] < N:
                        v = tree_dictionary[v][0]
                    point_array.append(v)
                    if v >= N:
                        node_q.extend(tree_dictionary[v])
                else:
                    print("???\n\n\n")
                    print(node_q.popleft())

            node_q.popleft()
            if len(node_q):
                node_q.append(None)
        # gather cluster values
        clusters.append(gd[torch.tensor(point_array)])
    return clusters

def template_level_order_traversal(tree_dictionary, tree_idx, global_data, N, k):
    # create numpy array for indexing
    gd = torch.tensor(global_data)
    # initialize node queue with initial indices
    node_q = deque([N + i for i in range(k)])
    node_q.append(None)
    # level counter
    lc = 0
    # perform level order traversal
    while len(node_q):
        # storage array for level values
        # NOTE: NOT storage for images. not immediately.
        level_vals = []
        while node_q[0] != None:
            # make sure node is actually a real node
            if node_q[0] in tree_dictionary:
                val = node_q.popleft()
                # if node is a direct path to a leaf, go to the leaf
                if len(tree_dictionary[v]) == 1 and tree_dictionary[v][0] < N:
                    v = tree_dictionary[v][0]
                level_vals.extend([v])
            else:
                # bypass any weird phantom nodes
                pass
                # print("???\n\n\n")
                # print(node_q.popleft())
        lc += 1
        # print(f"{lc}-{level_vals}")
        # init storage array for images
        images = []
        # populate image array with actual images(centroids and leaves)
        # if len(level_vals):
        #     images.extend(gd[torch.tensor(level_vals)])
        # print(images)
        # add new references to node q if they are not phantom nodes
        for i in level_vals:
            if i in tree_dictionary:
                node_q.extend(tree_dictionary[i])
        # handle end of level
        node_q.popleft()
        if len(node_q):
            node_q.append(None)

