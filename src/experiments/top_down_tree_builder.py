from sklearn.cluster import KMeans
import numpy as np
from collections import deque

def get_data_terms(data, k=2):
    kterms = np.arange(15)
    terms = []
    i = 0
    a = 1
    while i == 0 or terms[-1] <= len(data):
        terms.extend([np.sum(a * k ** kterms[0 : i + 1])])
        i += 1
    terms.extend([np.sum(a * k ** kterms[0 : i + 1])])
    return terms


def scatter_tree_build(data, k=2):
    # TODO: preprocess data
    data = np.array([i.ravel() for i in data])
    # print(len(data))
    o_data_indices = [i for i in range(len(data))]

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
    global_data[0 : 0 + len(data)] = data

    base_offt = len(data)

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

            # short circuit if one of the first data_indices is less than k
            if len(data_indices) <= k:
                # assign leaf indices as negative numbers to indicate special status
                for i in range(len(data_indices)):
                    tree_idx[wstart + i] = -1 * (data_indices[i] + 1)
                continue

            # perform kmeans
            kmeans = KMeans(n_clusters=k, init="k-means++")
            labels = kmeans.fit_predict(data[data_indices])

            # assign cluster centers and their indices to global_data and
            # tree_idx respectively
            for i, c in enumerate(kmeans.cluster_centers_):
                global_data[wstart + i] = c
                tree_idx[wstart + i] = i
            # collect cluster members
            associated_data = [deque() for _ in range(k)]
            for i, elem in enumerate(labels):
                associated_data[elem].append(data_indices[i])

            # handle each cluster
            for i, elem in enumerate(associated_data):
                # special case for when the cluster is less than k
                if len(elem) <= k:
                    # same as above except we are looking down into the next row
                    _base_addr = sum(row_lens[0 : curr_level + 1]) + base_offt
                    _block_posn = (block_posn + i) * k
                    _wstart = _base_addr + _block_posn
                    # we may need to realloc.
                    # TODO: this can introduce bad behaviors, should be looked at carefully
                    if _wstart + len(elem) >= len(tree_idx):
                        tree_idx.extend([None] * (_wstart+len(elem) - len(tree_idx)))
                        global_data.extend([None] * (len(tree_idx) - len(global_data)))
                    # assign leaf indices as negative numbers to indicate special status
                    for j, e in enumerate(elem):
                        tree_idx[_wstart + j] = -1 * (elem[j] + 1)
                    continue
                else:
                    # push the necessary things onto queues
                    local_block_q.extend([block_posn + i])
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

    return tree_idx, global_data, kt, row_lens


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
                    tree_dict[wstart + curr_idx] = []
                    tree_dict[wstart + curr_idx].extend(child_indices)

        # handle end of level
        local_block_q.popleft()
        base_addr_q.popleft()
        if len(local_block_q):
            local_block_q.append(None)
            base_addr_q.append(None)
    return tree_dict

def check_count(arr):
  count = 0
  for i in range(len(arr)):
    if type(arr[i]) == type(None):
      continue
    count += 1
  return count

def level_order_traversal(tree_dictionary, tree_idx, global_data,N,k):
  gd = np.array(global_data)
  tl = tree_dictionary
  print(gd)
  node_q = deque([N+i for i in range(k)])
  node_q.append(None)
  lc = 0
  while len(node_q):
    lvals = []
    while node_q[0] != None:
      if node_q[0] in tree_dictionary:
        v = node_q.popleft()
        if len(tree_dictionary[v]) == 1 and tree_dictionary[v][0] < N:
          v = tree_dictionary[v][0]
        lvals.extend([v])
      else:
        print("???\n\n\n")
        print(node_q.popleft())
        

    lc += 1
    print(f'{lc}-{lvals}')
    images = []    
    if len(lvals):
      images.extend(gd[np.array(lvals)])
    print(images)
    for i in lvals:
      if i in tree_dictionary:
        node_q.extend(tree_dictionary[i])
    node_q.popleft()
    if len(node_q):
      node_q.append(None)
  

def main():
    N = 200000
    data = np.random.randn(N, 2)
    k=4
    tree_idx, global_data, kt, row_lens = scatter_tree_build(data,k)
    tree_dictionary = gather_tree_dictionary(tree_idx, global_data, N, k, kt, row_lens)
    print(f'Valid_CHECK: {sum(tree_idx[0:N])}')
    d1 = check_count(tree_idx[N:])
    d2 = check_count(global_data[N:])
    treesum = len(tree_dictionary) + sum([len([i for i in v if i > N]) for k,v in tree_dictionary.items()])
    print(f'TREE NODES: {treesum}')
    print(f'TREE REF ARRAY: {d1}/{len(tree_idx[N:])}  @ {d1/len(tree_idx[N:])} density')
    print(f'TREE DATA ARRAY:{d2}/{len(global_data[N:])} @ {d2/len(global_data[N:])} density')
    # print(f'+{"="*50}')
    # for i,e in enumerate(global_data):
    #   if i == N:
    #     print(f'+{"="*50}')

    #   print(f'| {i}:\t{tree_idx[i]}\t{global_data[i]}')
    # print(tree_dictionary)
    # level_order_traversal(tree_dictionary, tree_idx,global_data, N,k)


if __name__ == "__main__":
    main()
