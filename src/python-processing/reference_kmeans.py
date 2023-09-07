# from clustering_imports import *
# from kmedoids import *
# from collections import deque
from clustering_driver import *


def weighted_sample(weights):
    """
    Sample a weighted probability distribution
    returns an index
    """
    # normalize total_w to be in [0,1]
    total_w = weights / np.sum(weights)
    sample_val = np.random.uniform(0, 1)
    for idx, w in enumerate(total_w):
        sample_val -= w
        if sample_val <= 0:
            return idx
    return len(weights) - 1


def kmeanspp(data_store, drefs, k):
    # k meanscpp
    dlen = len(drefs)
    start_center = np.random.randint(dlen)
    not_chosen = deque(i for i in range(dlen) if i != start_center)

    centroids = [data_store[drefs[start_center]]]

    chosen = {start_center}
    weights = None  # np.zeros(dlen)
    min_dist = float("inf")

    for _ in range(k - 1):
        weights = np.zeros(len(not_chosen))
        for idx, mdx in enumerate(not_chosen):
            min_dist = float("inf")
            m = data_store[drefs[mdx]]
            for ctx, ctr in enumerate(centroids):
                min_dist = min(min_dist, np.linalg.norm(m - ctr))

            weights[idx] = np.square(min_dist)

        selected_point = weighted_sample(weights)
        centroids.append(data_store[drefs[not_chosen[selected_point]]])
        chosen.add(not_chosen[selected_point])
        not_chosen.remove(not_chosen[selected_point])
    centroids = np.array(centroids)
    return centroids


def kmeans_main(data_store, drefs, centroids, FIRST_FLAG=False):
    # FIRST_FLAG = True
    dref_clusters = [[] for _ in centroids]
    dref_means = [np.zeros(data_store[0].shape) for _ in centroids]
    nn = None
    pdist = None
    min_dist = float("inf")

    for dref_idx, dref in enumerate(drefs):
        min_dist = float("inf")
        nn = None
        for ctx, ctr in enumerate(centroids):
            if FIRST_FLAG:
                if np.array_equal(data_store[dref], ctr):
                    continue

            pdist = np.linalg.norm(data_store[dref] - ctr)
            if pdist < min_dist:
                min_dist = pdist
                nn = ctx
        if nn != None:
            dref_clusters[nn].append(drefs[dref_idx])
            dref_means[nn] += data_store[drefs[dref_idx]]

    new_dref_clusters = []
    new_centroids = []
    for i in range(len(dref_clusters)):
        if not len(dref_clusters[i]):
            continue
        new_dref_clusters.append(dref_clusters[i])
        new_centroids.append(dref_means[i] / len(dref_clusters[i]))
    return new_dref_clusters, np.array(new_centroids)


def construct_tree(M, P, k=3, R=30, C=-1):
    data_store = M
    param_store = P
    node_list = []
    node_queue = deque()
    dref_queue = deque()

    node_list.append(ClusterTreeNode(0))
    node_queue.append(0)
    dref_queue.append([i for i in range(len(M))])

    while len(node_queue):
        node = node_list[node_queue.popleft()]
        drefs = dref_queue.popleft()

        node.children = []

        if len(drefs) > C:
            dref_clusters = None
            centroids = kmeanspp(data_store, drefs, k)

            for r in range(R):
                dref_clusters, new_centroids = kmeans_main(
                    data_store, drefs, centroids, bool(r == 0)
                )

                if (
                    new_centroids.shape != centroids.shape
                    or np.linalg.norm(new_centroids - centroids) == 0.0
                ):
                    centroids = new_centroids
                    break
                centroids = new_centroids

            for i, ctr in enumerate(centroids):
                idx = len(node_list)
                node.children.append(idx)
                node_list.append(ClusterTreeNode(ctr))
                node_queue.append(idx)
                dref_queue.append(dref_clusters[i])
        else:
            param_clusters, medioids, clusters, params, mlist = (
                None,
                None,
                None,
                None,
                None,
            )

            dlen = len(drefs)
            if not dlen:
                continue

            data = np.array([data_store[dref] for dref in drefs])
            if dlen > 1:
                mlist, distances = preprocess(data, k)
                total_sum = float("inf")

                for _ in range(R):
                    dref_clusters = assign_clusters(dlen, mlist, distances)
                    mlist = update_medioids(dref_clusters, mlist, distances)
                    new_sum = calculate_sum(dref_clusters, mlist, distances)
                    if new_sum == total_sum:
                        break
                    total_sum = new_sum

                medioids = np.array([data_store[mdref] for mdref in mlist])
                clusters = []
                param_clusters = []
                for i in range(len(mlist)):
                    param_clusters.append(
                        [param_store[dref] for dref in dref_clusters[i]]
                    )
                    clusters.append(
                        np.stack(
                            np.array([data_store[dref] for dref in dref_clusters[i]])
                        )
                    )

            elif dlen == 1:
                medioids = np.array([data[0]])
                clusters = [np.array([data[0]])]
                param_clusters = [[param_store[drefs[0]]]]

            for i, med in enumerate(medioids):
                idx = len(node_list)
                node.children.append(idx)
                node_list.append(ClusterTreeNode(med))
                node_list[idx].data_refs = np.array(clusters[i])
                node_list[idx].param_refs = param_clusters[i]

    return node_list
