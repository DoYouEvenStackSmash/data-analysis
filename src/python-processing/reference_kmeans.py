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


def kmeanspp(data_store, data_ref_arr, k):
    # k meanscpp
    dlen = len(data_ref_arr)
    start_center = np.random.randint(dlen)
    not_chosen = deque(i for i in range(dlen) if i != start_center)

    centroids = [data_store[data_ref_arr[start_center]]]

    chosen = {start_center}
    weights = None  # np.zeros(dlen)
    min_dist = float("inf")

    for _ in range(k - 1):
        weights = np.zeros(len(not_chosen))
        for idx, mdx in enumerate(not_chosen):
            min_dist = float("inf")
            m = data_store[data_ref_arr[mdx]]
            for ctx, ctr in enumerate(centroids):
                min_dist = min(min_dist, np.linalg.norm(m - ctr))

            weights[idx] = np.square(min_dist)

        selected_point = weighted_sample(weights)
        centroids.append(data_store[data_ref_arr[not_chosen[selected_point]]])
        chosen.add(not_chosen[selected_point])
        not_chosen.remove(not_chosen[selected_point])
    centroids = np.array(centroids)
    return centroids


def kmeans_main(data_store, data_ref_arr, centroids, FIRST_FLAG=False):
    """
    Perform K-Means clustering iterations for a subset of data references.

    Args:
        data_store (list of numpy arrays): Data points to be clustered.
        data_ref_arr (list of ints): Indices referring to elements in data_store.
        centroids (list of numpy arrays): Initial cluster centroids.
        FIRST_FLAG (bool): Flag to indicate the first iteration (default False).

    Returns:
        new_data_ref_clusters (list of lists): Updated cluster assignments for data references.
        new_centroids (numpy array): Updated cluster centroids.
    """

    data_ref_clusters = [[] for _ in centroids]
    data_ref_means = [np.zeros(data_store[0].shape) for _ in centroids]
    nn = None
    pdist = None
    min_dist = float("inf")

    for dref_idx, dref in enumerate(data_ref_arr):
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
            data_ref_clusters[nn].append(data_ref_arr[dref_idx])

    new_data_ref_clusters = []
    new_centroids = []
    for i in range(len(data_ref_clusters)):
        if not len(data_ref_clusters[i]):
            continue
        new_data_ref_clusters.append(data_ref_clusters[i])
        new_centroids.append(
            np.mean([data_store[i] for i in new_data_ref_clusters[-1]], axis=0)
        )

    return new_data_ref_clusters, np.array(new_centroids)


def construct_tree(M, P, k=3, R=30, C=-1):
    """
    Builds a hierarchical clustering on the input data M
    Returns a flat tree as a list of nodes, where node_list[0] is the root.
    """
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
        data_ref_arr = dref_queue.popleft()

        node.children = []

        if len(data_ref_arr) > C:
            data_ref_clusters = None
            centroids = kmeanspp(data_store, data_ref_arr, k)

            for r in range(R):
                data_ref_clusters, new_centroids = kmeans_main(
                    data_store, data_ref_arr, centroids, bool(r == 0)
                )

                if (
                    new_centroids.shape != centroids.shape
                    or np.linalg.norm(new_centroids - centroids) == 0.0
                ):
                    centroids = new_centroids
                    break
                centroids = new_centroids

            # create new nodes for centroids, and add each centroid/data pair to queues
            for i, ctr in enumerate(centroids):
                idx = len(node_list)
                node.children.append(idx)
                node_list.append(ClusterTreeNode(ctr))
                node_queue.append(idx)
                dref_queue.append(data_ref_clusters[i])

        # perform k medioids clustering to ensure that the center is within the input data
        else:
            param_clusters, medioids, clusters, params, mlist = (
                None,
                None,
                None,
                None,
                None,
            )

            dlen = len(data_ref_arr)
            if not dlen:
                continue

            data = np.array([data_store[dref] for dref in data_ref_arr])
            if dlen > 1:
                mlist, distances = preprocess(data, k)
                total_sum = float("inf")

                for _ in range(R):
                    data_ref_clusters = assign_clusters(dlen, mlist, distances)
                    mlist = update_medioids(data_ref_clusters, mlist, distances)
                    new_sum = calculate_sum(data_ref_clusters, mlist, distances)
                    if new_sum == total_sum:
                        break
                    total_sum = new_sum

                medioids = np.array([data_store[mdref] for mdref in mlist])

                param_clusters = [
                    [param_store[dref] for dref in cluster]
                    for cluster in data_ref_clusters
                ]

                clusters = [
                    np.stack(np.array([data_store[dref] for dref in cluster]))
                    for cluster in data_ref_clusters
                ]

            elif dlen == 1:
                medioids = np.array([data[0]])
                clusters = [np.array([data[0]])]
                param_clusters = [[param_store[data_ref_arr[0]]]]

            for i, med in enumerate(medioids):
                idx = len(node_list)
                node.children.append(idx)
                node_list.append(ClusterTreeNode(med))
                node_list[idx].data_refs = np.array(clusters[i])
                node_list[idx].param_refs = param_clusters[i]

    return node_list
