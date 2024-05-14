from sklearn.cluster import KMeans
import numpy as np


def get_data_size(data, k=2):
    d = 1
    while k ** d < len(data):
        d += 1
    sd = k ** d + k
    while k ** d > 1:
        d -= 1
        sd += k ** d
    return sd


def get_data_terms(data, k=2):
    d = 2
    while k ** (d - 2) <= len(data):
        d += 1
    kterms = np.arange(d)
    terms = [0] * d
    a = 1
    for i in range(d):
        terms[i] = np.sum(a * k ** kterms[0 : i + 1])
    return terms


from collections import deque


def tree_build(N, k=2):

    data = np.random.randn(N, 2) * 100
    data = np.array([i.ravel() for i in data])
    print(len(data))
    o_data_indices = [i for i in range(N)]
    # kmeans = KMeans(n_clusters=k)

    # print(data[data_indices])
    # labels = kmeans.fit_predict(data[o_data_indices])
    # associated_data = [[] for _ in range(k)]
    # for i,elem in enumerate(labels):
    #   associated_data[elem].append(o_data_indices[i])

    new_data_q = deque([i for i in range(1)])
    base_q = deque([0 for _ in range(1)])
    new_data_q.append(None)
    base_q.append(None)
    # for i in range(k):
    #   base_q.append(1)
    #   new_data_q.append(i)
    # base_q.append(None)
    # new_data_q.append(None)

    data_q = deque([o_data_indices])

    # for i,elem in enumerate(associated_data):
    #   data_q.append(associated_data[i])
    data_q.append(None)
    kt = get_data_terms(data, k)
    row_lens = [(kt[i + 1] - kt[i]) for i in range(len(kt) - 1)]
    print(sum(row_lens[0:1]))
    print(kt)
    global_data = [None] * (len(data) + (kt[-1]))  # - len(data)))
    tree_idx = [None] * (len(data) + (kt[-1]))  # - len(data)))
    for i, d in enumerate(data):
        global_data[i] = data[i]
    # print(len(global_data))
    base_offt = len(data)
    # print(len(global_data), len(tree_idx))
    # tree_idx[len(data):len(data)+k] = [i for i in range(k)]
    # base_addr_terms = get_data_terms(data, k)

    # depth = 1
    # print(base_offt)
    while len(new_data_q):
        while new_data_q[0] != None:

            data_indices = data_q.popleft()
            print(f"flag:{len(data_indices)}")

            parent_idx = new_data_q.popleft()

            curr_level = base_q.popleft()

            wbin = (parent_idx) * k
            base_addr = 0

            base_addr = base_offt + sum(row_lens[0:curr_level])

            wstart = base_addr + wbin
            wend = min(wstart + k, len(global_data))
            print(wstart)
            if len(data_indices) <= k:
                pass
                for i in range(len(data_indices)):
                    tree_idx[wstart + i] = -1 * (data_indices[i] + 1)
                continue

            kmeans = KMeans(n_clusters=k, init="k-means++")
            labels = kmeans.fit_predict(data[data_indices])
            # centroids = kmeans.cluster_centers_
            tree_start = wstart
            # labels = kmeans.fit_predict(data[data_indices])
            if wstart + len(kmeans.cluster_centers_) >= len(global_data):
                global_data.extend(
                    [None]
                    * (wstart + len(kmeans.cluster_centers_) - len(global_data) + k)
                )
            for i, c in enumerate(kmeans.cluster_centers_):
                global_data[wstart + i] = c
                tree_idx[tree_start + i] = i
            associated_data = [[] for _ in range(k)]

            for i, elem in enumerate(labels):
                associated_data[elem].append(data_indices[i])
            for i, elem in enumerate(associated_data):

                if len(elem) <= k:
                    # tree_idx[tree_start + i] = i
                    _base_addr = sum(row_lens[0 : curr_level + 1]) + base_offt
                    _wbin = (wbin + i) * k
                    _wstart = _base_addr + _wbin
                    if _wstart > 3000:
                        pass
                    print(_wstart)
                    if _wstart + len(elem) >= len(tree_idx):
                        tree_idx.extend(
                            [None] * ((_wstart + len(elem)) - len(tree_idx) + k)
                        )
                    for j, e in enumerate(elem):
                        tree_idx[_wstart + j] = -1 * (elem[j] + 1)
                    continue

                    # tree_idx[wstart + ]

                # if curr_level + 1 < len(row_lens):
                new_data_q.extend([wbin + i])
                base_q.extend([curr_level + 1])
                # print(elem)
                data_q.extend([associated_data[i]])

        new_data_q.popleft()
        base_q.popleft()
        data_q.popleft()
        if len(new_data_q):
            new_data_q.append(None)
            base_q.append(None)
            data_q.append(None)

    return tree_idx, global_data, kt, row_lens


def level_order_traversal(tree_idx, global_data, base_offt, k, kt, row_lens):

    new_data_q = deque([i for i in range(2)])
    new_data_q.append(None)
    base_q = deque([0 for _ in range(2)])
    base_q.append(None)
    depth = 1
    tree_dict = {}
    next_addr = lambda curr_level, wbin, i: sum(row_lens[0])
    # print(base_str)
    while len(new_data_q):
        while new_data_q[0] != None:

            # data_indices = data_q.popleft()

            parent_idx = new_data_q.popleft()

            curr_level = base_q.popleft()

            wbin = (parent_idx) * k
            base_addr = sum(row_lens[0:curr_level]) + base_offt
            wstart = base_addr + wbin
            wend = wstart + k

            # win_end = min(len(glowstart + k
            # print(wstart)
            for i, e in enumerate(tree_idx[wstart:wend]):
                child_idx = []
                if e == None:
                    continue
                if e < 0:
                    child_idx.append(-1 * (e + 1))

                    # print("DATA" + f'{wstart+i}')
                    # tree_idx[wstart+i] =
                    print(f"{wstart+i}->DATA{-1*(e+1)};")
                    # tree_idx[-1 * (e + 1)] = 1 if tree_idx[-1 * (e + 1)] == None else tree_idx[-1 * (e + 1)] + 1
                else:
                    for x in range(k):
                        addr = (
                            sum(row_lens[0 : curr_level + 1])
                            + base_offt
                            + (wbin + i) * k
                            + x
                        )
                        if tree_idx[addr] != None:
                            child_idx.append(
                                sum(row_lens[0 : curr_level + 1])
                                + base_offt
                                + (wbin + i) * k
                                + x
                            )
                    #   # new_data_q.extend([i])
                    #   # base_q.extend([next_base])
                    # else:
                    # tree_idx[wstart+i] = f'{wstart+i}->{base_offt + (row_lens[curr_level+1] if curr_level + 1 < len(row_lens) else sum(row_lens[0:curr_level])) + (wbin+i)*k+1};'
                    # tree_idx_str = ""
                    # for x in range(k):
                    #   addr = sum(row_lens[0:curr_level+1])+ (wbin+i)*k + x
                    #   if addr > len(global_data):
                    #     continue
                    #   if tree_idx[addr] == None:
                    #     continue
                    #   tree_idx_str = tree_idx_str + f'{wstart+i}->{addr};'
                    # if len(tree_idx_str):
                    #   print(tree_idx_str)

                    new_data_q.extend([wbin + i])
                    base_q.extend([curr_level + 1])
                if len(child_idx) and wstart + i not in tree_dict:
                    tree_dict[wstart + i] = []
                    tree_dict[wstart + i].extend(child_idx)
        # depth+=1
        new_data_q.popleft()
        base_q.popleft()
        if len(new_data_q):
            new_data_q.append(None)
            base_q.append(None)
    return tree_dict


def main():
    tb, gd, kt, rl = tree_build(9, 2)
    print(len(tb))
    # print(terms)
    td = level_order_traversal(tb, gd, 9, 2, kt, rl)
    print(td)
    # print(tb)
    # print(global_data)
    print(f'+{"="*50}')
    for i, e in enumerate(gd):
        if i == 9:
            print(f'+{"="*50}')

        print(f"| {i}:\t{tb[i]}\t{gd[i]}")

    print(len(tb))


if __name__ == "__main__":
    main()
