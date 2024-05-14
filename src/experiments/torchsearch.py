import torch
import numpy as np
def calculate_distance(T, S, noise):
    return torch.sqrt(torch.sum(((T - S) ** 2),dim=(-2,-1)))

from collections import *
def prep(global_data,radius_data):
    # gd = torch.stack([i if i != None else torch.empty(global_data[0].shape) for i in global_data])
    # print(gd.shape)
    rd = torch.stack([i if i != None else torch.tensor(float('nan')) for i in radius_data ])
    return rd


def level_order_search(T,tree_dictionary, radius_data, global_data,ctf, N, k, tau):
    # create numpy array for indexing
    # print(T)
    # T = T.cuda()
    # ctf = ctf.cuda()

    results = deque()
    # gd = np.array(global_data,dtype=object)
    # print(type(global_data))
    gd = global_data#[i if i != None else torch.zeros_like(global_data[0]) for i in global_data]# = torch.tensor(float('nan'))
    rd = radius_data#[i if i != None else torch.tensor(float('nan')) for i in radius_data ]# = torch.tensor(float('nan'))
    T = T.cuda()
    ctf = ctf.cuda()
    # print(gd)
    # gd = torch.stack(gd)
    # rd = torch.stack(rd)
    
    # gd= torch.tensor([float(i) for i in gd], dtype=torch.float64)
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
            # make sure node is actually a real node
            if node_q[0] in tree_dictionary:
                val = node_q.popleft()
                # if node is a direct path to a leaf, go to the leaf
                if len(tree_dictionary[val]) == 1 and tree_dictionary[val][0] < N:
                    v = tree_dictionary[val][0]
                level_vals.extend([val])
            else:
                if node_q[0] < N:
                    level_vals.extend([node_q.popleft()])
                else:
                    # bypass any weird phantom nodes
                    pass
                    # print("???\n\n\n")
                    node_q.popleft()
        lc += 1
        # print(f"{lc}-{level_vals}")
        # init storage array for images
        images = []
        # populate image array with actual images(centroids and leaves)
        if len(level_vals):
            for i,elem in enumerate(level_vals):
                # print(elem)
                # print(gd)
                # print(gd[elem])
                if torch.isnan(gd[elem][0]):
                    level_vals[i] = tree_dictionary[level_vals[i]][0]

            images = gd[torch.tensor(level_vals)].cuda()
            distances = calculate_distance(T.expand_as(images), images*ctf.expand_as(images),0).cpu()
            print(distances)
            # torch.cuda.empty_cache()
            # print(f'DIST{distances}')
            # print('\n')
        # print(images)
        # add new references to node q if they are not phantom nodes
            for i,elem in enumerate(distances):
                # print(elem)
                if level_vals[i] >= N and elem > rd[level_vals[i]]+tau:
                    continue
                if level_vals[i] in tree_dictionary:
                    node_q.extend(tree_dictionary[level_vals[i]])
                if level_vals[i] < N and elem < tau:
                    results.append(level_vals[i])
        # handle end of level
        node_q.popleft()
        if len(node_q):
            node_q.append(None)
    return results

import torch
import numpy as np

from collections import *
def prep(global_data,radius_data):
    # gd = torch.stack([i if i != None else torch.empty(global_data[0].shape) for i in global_data])
    # print(gd.shape)
    rd = torch.stack([i if i != None else torch.tensor(float('nan')) for i in radius_data ])
    return rd

mag = lambda x: torch.sqrt(x.real**2 + x.imag**2)

def compact_level_order_search(T,tree_dictionary, radius_data, global_data,ctf, N, k, tau):
    # create numpy array for indexing
    # print(T)
    # T = T.cuda()
    # ctf = ctf.cuda()

    results = deque()
    # gd = np.array(global_data,dtype=object)
    # print(type(global_data))
    gd = global_data#[i if i != None else torch.zeros_like(global_data[0]) for i in global_data]# = torch.tensor(float('nan'))
    rd = radius_data#[i if i != None else torch.tensor(float('nan')) for i in radius_data ]# = torch.tensor(float('nan'))
    T = T.cuda()
    ctf = ctf.cuda()
    # print(gd)
    # gd = torch.stack(gd)
    # rd = torch.stack(rd)
    
    # gd= torch.tensor([float(i) for i in gd], dtype=torch.float64)
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
            
            level_vals.extend(node_q)
            level_vals.pop()
            node_q = deque([None])
            
            # # make sure node is actually a real node
            # if node_q[0] in tree_dictionary:
            #     # val = node_q.popleft()
            #     # # if node is a direct path to a leaf, go to the leaf
            #     # if len(tree_dictionary[val]) == 1 and tree_dictionary[val][0] < N:
            #     #     v = tree_dictionary[val][0]
            #     level_vals.extend([node_q.popleft()])
            # else:
            #     if node_q[0] < N:
            #         level_vals.extend([node_q.popleft()])
            #     else:
            #         # bypass any weird phantom nodes
            #         pass
            #         # print("???\n\n\n")
            #         node_q.popleft()
        lc += 1
        # print(f"{lc}-{level_vals}")
        # init storage array for images
        images = []
        # populate image array with actual images(centroids and leaves)
        if len(level_vals):
            # for i,elem in enumerate(level_vals):
            #     # print(elem)
            #     # print(gd)
            #     # print(gd[elem])
            #     if torch.isnan(gd[elem][0]):
            #         level_vals[i] = tree_dictionary[level_vals[i]][0]

            images = gd[torch.tensor(level_vals)].cuda()
            Tp = T.expand_as(images)
            Cp = ctf.expand_as(images).cuda()
            Ip = images * Cp
            # print(Tp.shape,Cp.shape)
            # print(Ip.shape)
            distances = mag(calculate_distance(Tp, Ip,0)).cpu()
            # print(distances.shape)
            # exit()
            # torch.cuda.empty_cache()
            # print(f'DIST{distances}')
            # print('\n')
        # print(images)
        # add new references to node q if they are not phantom nodes
            for i,elem in enumerate(distances):
                # print(elem)
                print(type(elem))
                if level_vals[i] >= N and mag(elem) > rd[level_vals[i]]+tau:
                    continue
                if level_vals[i] in tree_dictionary:
                    node_q.extend(tree_dictionary[level_vals[i]])
                if level_vals[i] < N and mag(elem) < tau:
                    results.append(level_vals[i])
        # handle end of level
        node_q.popleft()
        if len(node_q):
            node_q.append(None)
    return results