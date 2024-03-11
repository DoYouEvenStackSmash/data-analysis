# from clustering_imports import *
# from kmedoids import *
# from collections import deque
from kmedoids import *
from kmeans import *
from clustering_driver import *
from sklearn.cluster import KMeans

SKLEARN_KMEANS = True


# import logging
# sklearn_logger = logging.getLogger('sklearnex')
# sklearn_logger.setLevel(logging.ERROR)
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
                    if jnp.linalg.norm(img.m1 - node.val.m1) == 0.0:
                        node.val_idx = img_count
                    data_list[img_count] = img.m1  # .numpy()
                    # param_list[img_count] = node.param_refs[j]
                    data_idx.append(img_count)
                    # param_idx.append(img_count)
                    img_count += 1
                node_list[i + 1].data_refs = data_idx
                # node_list[i + 1].param_refs = param_idx
    # print(img_count)
    return data_list, param_list


CONST_X = 128
CONST_Y = 128


def construct_tree(M, k=3, R=30, C=1):
    """
    Builds a hierarchical clustering on the input data M
    Returns a flat tree as a list of nodes, where node_list[0] is the root.
    """
    # if k == 2:
    #     return balltree_wrapper(M, C)
    data_store = M
    max_filter = M[0].m2
    node_list = []
    node_queue = deque()
    dref_queue = deque()
    rootd = DatumT()
    rootd.m1 = np.zeros((CONST_X, CONST_Y))
    rootd.m2 = M[0].m2
    # dst = np.array(
    #     [data_store[i].m1.astype(jnp.float32).ravel() for i in range(len(data_store))]
    # )
    # rootd.m1 = np.mean(dst)
    node_list.append(ClusterTreeNode(rootd))
    node_queue.append(0)
    dref_queue.append([i for i in range(len(M))])
    # distances = []
    while len(node_queue):
        node = node_list[node_queue.popleft()]
        data_ref_arr = dref_queue.popleft()

        node.children = []

        if len(data_ref_arr) >= C:  # and len(data_ref_arr) > k:
            centroids = None
            data_ref_clusters = None
            if SKLEARN_KMEANS:
                kmeans = KMeans(n_clusters=min(k, len(data_ref_arr)), init="k-means++")
                dst = np.array(
                    [
                        jax_apply_d1m2_to_d2m1(data_store[i], data_store[i])
                        .astype(jnp.float64)
                        .ravel()
                        for i in data_ref_arr
                    ]
                )

                labels = kmeans.fit_predict(dst)
                centroids = kmeans.cluster_centers_
                # labels = kmeans.labels_
                data_ref_clusters = [[] for _ in range(min(k, len(data_ref_arr)))]
                ctx = [DatumT() for c in centroids]
                for i, c in enumerate(ctx):
                    ctx[i].m1 = centroids[i].reshape((CONST_X, CONST_Y))
                    ctx[i].m2 = max_filter
                centroids = ctx
                for i, j in enumerate(labels):
                    data_ref_clusters[j].append(data_ref_arr[i])
            #     # data_ref_clusters[j].append(data_ref_arr[i])
            else:
                data_ref_clusters = None
                centroids = kmeanspp_refs(data_store, data_ref_arr, k)
                new_centroids = None

                for r in range(R):
                    data_ref_clusters, new_centroids = kmeans_refs(
                        data_store, data_ref_arr, centroids, bool(r == 0)
                    )

                    if (
                        len(new_centroids) != len(centroids)
                        or jnp.allclose(
                            jnp.stack([c.m1 for c in centroids]),
                            jnp.stack([c.m1 for c in new_centroids]),
                        )
                        # len(new_centroids) != len(centroids)
                        # or jnp.linalg.norm(jnp.stack([centroids[i].m1 for i in range(len(centroids))]) - \
                        #     jnp.stack([new_centroids[i].m1 for i in range(len(new_centroids))])) == 0.0
                    ):
                        centroids = new_centroids
                        break
                    centroids = new_centroids

            # create new nodes for centroids, and add each centroid/data pair to queues
            node.children = [
                i for i in range(len(node_list), len(node_list) + len(centroids))
            ]

            nl = [ClusterTreeNode(ctr) for ctr in centroids]

            for i, d in enumerate(data_ref_clusters):
                if len(data_ref_clusters[i]) < C:
                    # create new node
                    nl[i].data_refs = [data_store[x] for x in data_ref_clusters[i]]
                    nl[i].children = None
                else:
                    node_queue.append(node.children[i])
                    dref_queue.append(data_ref_clusters[i])
            node_list.extend(nl)

            for x, i in enumerate(node.children):
                if len(data_ref_clusters[x]) > 1:
                    # dst = np.array(
                    #     [
                    #         jax_apply_d1m2_to_d2m1(data_store[j], data_store[j]).astype(jnp.float32).ravel()
                    #         for j in data_ref_clusters[x]
                    #     ]
                    # )
                    dst = [
                        jax_apply_d1m2_to_d2m1(data_store[j], data_store[j])
                        for j in data_ref_clusters[x]
                    ]
                    # node_list[i].cluster_radius = float(
                    #     jnp.sqrt(
                    #         jnp.sum((jax_apply_d1m2_to_d2m1(node_list[i].val, node_list[i].val).flatten() - dst) ** 2)
                    #     ).astype(jnp.float32)
                    # )
                    center = jax_apply_d1m2_to_d2m1(node_list[i].val, node_list[i].val)
                    for a, d in enumerate(dst):
                        node_list[i].cluster_radius = float(
                            max(
                                jnp.linalg.norm(center - d), node_list[i].cluster_radius
                            )
                        )

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
            # node.cluster_radius = 0
            if not dlen:
                continue
            # if dlen > 1:

            data = [data_store[dref] for dref in data_ref_arr]
            if dlen > k:
                mlist, distances = preprocess(data, k)
                total_sum = float("inf")
                data_ref_clusters = None
                new_sum = None
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
            # elif dlen == k:
            #     medioids = [data[i] for i in range(len(data))]
            #     clusters = [[data[i]] for i in range(len(data))]
            elif dlen > 1:
                # for _ in range(k):  # Assuming k is the number of clusters
                D = jnp.zeros(len(data_ref_arr))
                medioids = []
                for i in range(len(data_ref_arr)):
                    for j in range(len(data_ref_arr)):
                        D.at[i].set(
                            D[i]
                            + jnp.linalg.norm(
                                data_store[data_ref_arr[i]].m1
                                - data_store[data_ref_arr[j]].m1
                            )
                        )

                min_idx = jnp.argmin(D).item()
                medioids.append(data_store[min_idx])
                clusters = [[data_store[i] for i in data_ref_arr]]

            elif dlen == 1:
                medioids = [data[0]]
                clusters = [[data[0]]]
                node.val = data[0]
                node.data_refs = [data[0]]
                node.children = None
                node.cluster_radius = 0
                continue
            dstr = []
            for i, med in enumerate(medioids):
                idx = len(node_list)
                node.children.append(idx)
                node_list.append(ClusterTreeNode(med))
                node_list[idx].cluster_radius = 0
                dstr.extend([j.m1.astype(jnp.float32).ravel() for j in clusters[i]])
                if len(clusters[i]) > 1:
                    dst = np.array(
                        [
                            j.m1.astype(jnp.float32).ravel()
                            for j in clusters[i]
                            if j != med
                        ]
                    )
                    node_list[idx].cluster_radius = float(
                        jnp.sqrt(
                            jnp.sum((node_list[idx].val.m1.flatten() - dst) ** 2)
                        ).astype(jnp.float32)
                    )
                    # node_list.cluster_radius =
                node_list[idx].data_refs = clusters[i]
                node_list[idx].children = None

    node_list[0].cluster_radius = sum(
        [float(node_list[i].cluster_radius) for i in node_list[0].children]
    )
    return node_list
