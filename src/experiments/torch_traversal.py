import torch
import numpy as np

from collections import *
from aux_functions import *


def preprocess(exp_data, N, k):
    comparison_q = deque()
    result_dict = {}
    for x, e in enumerate(exp_data):
        result_dict[x] = deque()
        comparison_q.extend([torch.as_tensor([x, N + s],device=torch.device('cuda')) for s in range(k)])
    return comparison_q, result_dict
import time

def flush_traversal(
    exp_data,
    ctf_data,
    tree_dict,
    radius_data,
    global_data,
    N,
    k,
    tau,
    comparison_q,
    result_dict,
    batch_size=1,
):
    global_data = global_data.cuda()
    exp_data = exp_data.cuda()
    ctf_data = ctf_data.cuda()
    A = False
    EXP = 0
    SIM = 1
    print(f"{len(comparison_q)}QLEN")
    pre_comparison_q = deque()
    construction_q = deque()
    comparisons = 0
    # solutions_x = deque()
    # solutions_s = deque()
    solutions_sx = deque()
    distance_blocks = deque()
    chunks = deque()
    while len(comparison_q) or len(distance_blocks):
        # radius_data = radius_data.cuda()
        while len(comparison_q):
            
            sz = len(comparison_q)
            # chunk2 = torch.concat([comparison_q.popleft() for i in range(min(sz, batch_size))])
            chunk2 = torch.stack(
                [comparison_q.popleft() for i in range(min(sz, batch_size))], dim=1
            )        
            # mask1 = torch.isnan(radius_data[chunk2[SIM,:]])
            # chunk2 = chunk.cuda()
            chunks.append(chunk2)
            distance_blocks.append(calculate_distance(exp_data[chunk2[EXP, :]], global_data[chunk2[SIM, :]]*ctf_data[chunk2[EXP, :]], 0))
        while len(distance_blocks):
            # click on and click off
            mask1 = torch.isnan(radius_data[chunks[0][SIM,:]])
            mask2=distance_blocks[0] < tau
            
            
            # # all things with distances < 0
            # mask2 = distance_blocks[0] < tau
            
            # valid leaf nodes
            # valid_leaves = mask1 & mask2
            # if len(torch.nonzero(mask1)):
            # print(torch.nonzero(mask1 & mask2).shape)
            if torch.nonzero(mask1 & mask2).shape[0] != 0:
                solutions_sx.append(chunks[0][:,torch.nonzero(mask1 & mask2)])
            # print(solutions_sx)
            # solutions_x.append(chunk2[EXP,mask1 & mask2])
            # solutions_s.append(chunk2[SIM,mask1 & mask2])        
            # mask4 = ~mask1 & ~mask3
            # f = torch.tensor(tree_dict[chunk[SIM,torch.nonzero(mask3)].item()])
            # comparison_q.extend([torch.tensor(j) for j in zip(torch.tensor(chunk[EXP,i.cpu()]).expand_as(f),f)])
            # ft = torch.tensor([tree_dict[chunk2[SIM,i].item()] for i in torch.nonzero(mask3)])
            # cpt = torch.tensor(chunk[EXP, torch.nonzero(mask3)] )
            mask3 = (radius_data[chunks[0][SIM,:]] + tau > distance_blocks[0]) & ~mask1
            # print("MASK",torch.nonzero(mask3))
            if torch.nonzero(mask3).shape[0] != 0:
                pre_comparison_q.extend([[tree_dict[chunks[0][SIM,i].item()]] for i in torch.nonzero(mask3)])
                # construction_q.extend([chunks[0][EXP,i] for i in torch.nonzero(mask3)])
                construction_q.extend(chunks[0][EXP,mask3])# for i in torch.nonzero(mask3)])

            # for i in torch.nonzero((radius_data[chunks[0][SIM,:]] + tau > distance_blocks[0]) & ~mask1):
            #     # if radius_data[chunk2[SIM,i]] + tau < distances[i]:
            #     #     continue
            #     construction_q.extend([chunks[0][EXP,])
            #     comparison_q.extend([torch.as_tensor([chunks[0][EXP,i],n],device=torch.device('cuda')) for n in tree_dict[chunks[0][SIM,i].item()]])
            distance_blocks.popleft()
            chunks.popleft()
            # if len(comparison_q) > 120000:
            # break
        while len(construction_q):
            # print("CONSTRUCTION",construction_q)
            # print("COMPARISON",pre_comparison_q)
            # trickle
            simulated_range = torch.as_tensor(pre_comparison_q.popleft(),device=torch.device('cuda')).squeeze()
            experimental = torch.as_tensor(construction_q.popleft(),device=torch.device('cuda')).expand_as(simulated_range).squeeze()
            # print(experimental)
            # print(simulated_range)#.shape)
            # elem = [torch.as_tensor(i,device=torch.device('cuda')) for i in zip(experimental,simulated_range)]
            # elem = torch.stack([experimental,simulated_range],dim=1)
            # print(elem)
            # exit()
            # exit()
            # for e in experimental:
            comparison_q.extend([torch.as_tensor(i,device=torch.device('cuda')) for i in zip(experimental,simulated_range)])
            # experimental = torch.as_tensor(construction_q.popleft(),device=torch.device('cuda'))
            # simulated = pre_comparison_q.popleft()
            # print(simulated)
            # print(experimental)
            # exit()
            # comparison_q.append(torch.as_tensor([experimental.expand_as(simulated),simulated],device=torch.device('cuda')))
            # g = torch.tensor(chunk2[EXP,i]).expand_as(f)
            # comparison_q.extend(torch.stack([g,f]))
            # comparison_q.extend([torch.tensor(j) for j in zip(torch.tensor(chunk2[EXP,i]).expand_as(f),f)])
        # torch.cuda.empty_cache()
    print("Serializing")
    if A:
        solutions_x = torch.concat([i for i in solutions_x])
        solutions_s = torch.concat([i for i in solutions_s])
    
        distances = calculate_distance(exp_data[solutions_x.cpu()].cuda(),global_data[solutions_s.cpu()].cuda(), ctf_data[solutions_x.cpu()].cuda(),0)
        mask = distances < tau
        solutions_x = solutions_x.cpu()
        solutions_s = solutions_s.cpu()
        for i in torch.nonzero(mask.cpu()):
            result_dict[solutions_x[i].item()].append(solutions_s[i].item())
    else:
        print([i.shape for i in solutions_sx])
        solutions_sx = torch.concat([i.squeeze(-1) for i in solutions_sx],dim=-1)

        # exit()
        distances = calculate_distance(exp_data[solutions_sx[EXP,:]],global_data[solutions_sx[SIM,:]]*ctf_data[solutions_sx[EXP,:]],0)
        mask = distances < tau
        solutions_sx = solutions_sx.cpu()
        for i in torch.nonzero(mask.cpu()):
            result_dict[solutions_sx[EXP,i].item()].append(solutions_sx[SIM,i].item())
    
    # print(solutions_x)
    # for i in range(len(solutions_x)):
    #     distances = calculate_distance(exp_data[solutions_s[i].cpu()].cuda(),global_data[solutions_x[i].cpu()].cuda(),ctf_data[solutions_s[i].cpu()].cuda(),0)
    #     mask = distances < tau
        
    return result_dict


mag = lambda x: torch.sqrt(x.real**2 + x.imag**2)


def compact_level_order_search(
    T, tree_dictionary, radius_data, global_data, ctf, N, k, tau
):
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

            distances = calculate_distance(Tp, images, Cp, 0).cpu()

            # distances = mag(calculate_distance(Tp, images,Cp ,0)).cpu()

            # torch.cuda.empty_cache()
            # add new references to node q if they are not phantom nodes
            for i, elem in enumerate(distances):

                if level_vals[i] >= N and elem > radius_data[level_vals[i]] + tau:
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
    # puts all members of a cluster into the same array
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
        if elem < N:  # if elem < N, it's a leaf
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
