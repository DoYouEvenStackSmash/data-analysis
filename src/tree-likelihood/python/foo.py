def search_node(
    T, n_idx, node_list, data_list, dbox, dbest, nearest_neighbor, noise=1, depth=0
):
    global exclude_count
    if node_list[n_idx].data_refs != None:  # isLeaf = True
        for idx in node_list[n_idx].data_refs:
            search_leaf(T, idx, data_list, dbest, nearest_neighbor, noise)
        # return
    if (
        node_list[n_idx].data_refs == None and dbox < dbest[0]
    ):  # isLeaf = false and distance to boundary of cluster is less than the best distance so far
        distance_to_cluster_boundary = (
            [  # calculate the next cluster boundary distances
                (
                    node_list[n_idx].cluster_radius
                    - difference(T.m1, node_list[idx].val.m1, noise),
                    idx,
                )
                for i, idx in enumerate(node_list[n_idx].children)
            ]
        )

        sortkey = lambda x: -x[0]
        distance_to_cluster_boundary = sorted(
            distance_to_cluster_boundary, key=sortkey
        )  # sort the cluster boundary distances in decreasing order to account for unbalanced tree
        # given that we have sorted in descendign order, we know that the
        for idx, c in enumerate(
            distance_to_cluster_boundary
        ):  # for each child of the current cluster, search for target with dbox= distance between child and current cluster boundary
            search_node(
                T,
                distance_to_cluster_boundary[idx][1],
                node_list,
                data_list,
                dbox,
                dbest,
                nearest_neighbor,
                noise,
                depth + 1,
            )
            jdx = [
                rdx for rdx in range(len(distance_to_cluster_boundary)) if rdx != idx
            ]  # collect indices for other children where the boundary distance is less current child distance + sum of all other child distances
            for j in jdx:
                search_node(
                    T,
                    distance_to_cluster_boundary[j][1],
                    node_list,
                    data_list,
                    dbox
                    - distance_to_cluster_boundary[idx][0]
                    + sum(
                        [
                            distance_to_cluster_boundary[jrx][0]
                            for jrx in jdx
                            if jrx != j
                        ]
                    ),
                    dbest,
                    nearest_neighbor,
                    noise,
                    depth + 1,
                )
    elif dbox > dbest[0]:
        if depth not in exclude_count:
            exclude_count[depth] = 0
        exclude_count[depth] += 1
