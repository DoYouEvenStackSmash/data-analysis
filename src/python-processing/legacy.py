def preprocess_2D(M, k=3):
    """
    Preprocesses M according to k medioids algorithm
    Computes an initial set of medioids, and a distances matrix between every pair of points
    Returns a list of indices referring to M, and a distances matrix
    """
    n = M.shape[0]

    distances = np.sqrt(np.sum(np.square(M[:, np.newaxis, :] - M), axis=2))

    # Step 1-2: Calculate denominators efficiently
    denominators = np.sum(distances, axis=1)
    # Calculate v values using vectorized operations
    v_values = distances / denominators[:, np.newaxis]
    np.fill_diagonal(v_values, 0)  # Set diagonal values to 0
    v_sums = np.sum(v_values, axis=1)
    # Initialize objects using list comprehension
    data = [Datum(idx) for idx in range(n)]

    # Assign calculated v values to data objects
    for j in range(n):
        data[j].v = v_sums[j]
        # Sort the medioid indices by v values

    # medioid_indices = medioid_indices[np.argsort(v_sums[medioid_indices])]
    sortkey = lambda d: d.v
    sorted_data = sorted(data, key=sortkey)
    medioid_indices = np.argpartition(np.array([d.v for d in sorted_data]), k)[:k]
    mlist = list(medioid_indices)
    return mlist, distances


def nested_kmeans(M):
    """
    Builds a hierarchical clustering on input space M

    """
    k = 3  # Number of clusters for k-means
    max_iter = 30  # Maximum number of iterations for k-means
    start_center = np.random.randint(
        M.shape[0]
    )  # Choose a random starting center index
    centroids = [M[start_center]]  # Initialize centroids list with the starting center

    # Create a deque of indices for non-center points
    not_chosen = deque(i for i in range(len(M)) if i != start_center)
    chosen = {start_center}  # Set containing chosen center indices

    # Compute initial centroids using k-means++
    centroids = kmeanspp(M, k, centroids, not_chosen, chosen)

    cdict = {}  # Dictionary to store clusters at each level
    cq = deque()  # Queue for centroids
    dq = deque()  # Queue for data

    cq.append(centroids)  # Enqueue initial centroids
    dq.append(M)  # Enqueue the entire data array
    centroid_arr = []  # List to store all centroids

    cutoff = 65  # Cutoff point for switching to k-medioids
    while bool(cq):
        centroids = cq.popleft()  # Dequeue centroids
        data = dq.popleft()  # Dequeue data

        # If data size is below the cutoff, use k-medioids clustering
        if len(data) < cutoff:
            clusters, centroids = kmedioids(data, k, max_iter=30)
            for i, ctr in enumerate(centroids):
                ci = np.array(clusters[i])

                # Store the cluster data in cdict and the centroid in centroid_arr
                cdict[len(centroid_arr)] = ci
                centroid_arr.append(ctr)
            continue  # Skip further processing for this level

        else:
            # Apply regular k-means clustering
            for _ in range(max_iter):
                clusters = assign_kmeans_clusters(data, centroids)
                new_centroids = update_centroids(clusters)
                if np.array_equal(centroids, new_centroids):
                    break
                centroids = new_centroids

        # Process clusters and enqueue data for the next level
        for i, ctr in enumerate(centroids):
            ci = np.array(clusters[i])

            # Store the cluster data in cdict and the centroid in centroid_arr
            cdict[len(centroid_arr)] = ci
            centroid_arr.append(ctr)

            # Enqueue the centroid and cluster data for the next level
            cq.append(ctr)
            dq.append(ci)

    clusters = []
    centroids = []
    for k, v in cdict.items():
        centroids.append(centroid_arr[k])
        clusters.append(v)

    return clusters, centroids  # Return the final clusters and centroids


k = 3
M = dataloader(filename)
node_list = construct_tree(M, k)
for i, node in enumerate(node_list[1:]):
    if node.children is None:
        if node.data is not None:
            clusters.append(node.data)
            centroids.append(node.val)
            continue
        continue
display_clusters(clusters, np.array(centroids))
