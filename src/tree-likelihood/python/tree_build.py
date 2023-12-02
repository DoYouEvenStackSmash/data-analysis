# from clustering_imports import *
# from kmedoids import *
# from collections import deque
from kmedoids import *
from kmeans import *
from clustering_driver import *


def construct_data_list(node_list, data_shape=None):
    """
    Wrapper function for constructing a new data array by replacing data references to their indices
    Returns an array of data points, and replaces data_refs in leaf nodes
    """

    data_list = [np.empty(data_shape[1:]) for _ in range(data_shape[0])]
    param_list = [[] for _ in range(data_shape[0])]
    img_count = 0

    for i, node in enumerate(node_list[1:]):
        if node.children is None:
            if node.data_refs is not None:
                data_idx = []
                # param_idx = []
                for j, img in enumerate(node.data_refs):
                    if torch.linalg.norm(img.m1 - node.val.m1) == 0.0:
                        node.val_idx = img_count
                    data_list[img_count] = img.m1.numpy()
                    # param_list[img_count] = node.param_refs[j]
                    data_idx.append(img_count)
                    # param_idx.append(img_count)
                    img_count += 1
                node_list[i + 1].data_refs = data_idx
                # node_list[i + 1].param_refs = param_idx
    # print(img_count)
    return data_list, param_list


def construct_tree(M, k=3, R=30, C=-1):
    """
    Builds a hierarchical clustering on the input data M
    Returns a flat tree as a list of nodes, where node_list[0] is the root.
    """
    data_store = M
    # param_store = P
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
            centroids = kmeanspp_refs(data_store, data_ref_arr, k)
            for r in range(R):
                data_ref_clusters, new_centroids = kmeans_refs(
                    data_store, data_ref_arr, centroids, bool(r == 0)
                )
                
                if (
                    len(new_centroids) != len(centroids)
                    or torch.linalg.norm(torch.stack([centroids[i].m1 for i in range(len(centroids))]) - \
                        torch.stack([new_centroids[i].m1 for i in range(len(new_centroids))])) == 0.0
                ):
                    centroids = new_centroids
                    break
                centroids = new_centroids
            print(f"shorted: {len(centroids)}")
            # create new nodes for centroids, and add each centroid/data pair to queues
            for i, ctr in enumerate(centroids):
                idx = len(node_list)
                print(idx)
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

            data = [data_store[dref] for dref in data_ref_arr]
            if dlen >= k:
                mlist, distances = preprocess(data, k)
                total_sum = float("inf")

                for _ in range(R):
                    data_ref_clusters = assign_clusters(dlen, mlist, distances)
                    mlist = update_medioids(data_ref_clusters, mlist, distances)
                    new_sum = calculate_sum(data_ref_clusters, mlist, distances)
                    if new_sum == total_sum:
                        break
                    total_sum = new_sum

                medioids = [data_store[mdref] for mdref in mlist]


                clusters = [
                    [data_store[dref] for dref in cluster]
                    for cluster in data_ref_clusters
                ]

            elif dlen > 1:
                # for _ in range(k):  # Assuming k is the number of clusters
                D = torch.zeros(len(data_ref_arr))
                
                for i in range(len(data_ref_arr)):
                    for j in range(len(data_ref_arr)):
                        D[i] += torch.linalg.norm(data_store[data_ref_arr[i]].m1 - data_store[data_ref_arr[j]].m1)

                min_idx = torch.argmin(D).item()
                medioids.append(data_store[min_idx])
                clusters = [[data_store[i] for i in data_refs]]
                
            elif dlen == 1:
                medioids = [data[0]]
                clusters = [[data[0]]]
                # param_clusters = [[param_store[data_ref_arr[0]]]]

            for i, med in enumerate(medioids):
                idx = len(node_list)
                node.children.append(idx)
                node_list.append(ClusterTreeNode(med))
                node_list[idx].data_refs = clusters[i]
                # node_list[idx].param_refs = param_clusters[i]

    return node_list
