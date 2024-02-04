# target


def level_order_search(node_list, T, TAU=1e-8):
    LEVEL_FLAG = -1
    q.append(node_list[0])
    # q.append(-1)
    reachable_clusters = []
    level_counter = 0

    while len(q):
        q.append(LEVEL_FLAG)
        level_counter += 1
        reachable_clusters.append([])

        while not q[0] != LEVEL_FLAG:
            elem_id = q.pop(0)
            elem = node_list[elem_id]

            if elem.data_refs == None:
                for cidx in elem.children:
                    res_elem_tensor = apply_d1m2_to_d2m1(T, node_list[cidx].val)
                    dist = torch.linalg.norm(T.m1 - res_elem_tensor)
                    if dist > TAU:
                        continue
                    q.push_back(cidx)
            else:
                reachable_cluster_refs[-1].append(elem_id)

        flag = q.pop(0)

    return reachable_cluster_refs
