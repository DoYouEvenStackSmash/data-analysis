#!/usr/bin/python3


def D_vect(di, dj):
    """
    Euclidean distance between two elements
    """
    return np.sqrt(np.sum(np.square(di - dj), axis=0))


def sift_data(clusters, centroids, r):
    """
    Experimental behavior for creating a ball around the centroids for excluding outliers
    Returns a numpy array of the excluded points
    """
    data = []
    for i, k in enumerate(centroids):
        for idx, pt in enumerate(clusters[i]):
            if D_vect(k, pt) > r:
                data.append(np.array(clusters[i][idx]))
    return np.array(data)


def eval_cluster_inertia(clusters, centroids):
    """
    Calculates the inertia of each cluster
    """
    min_rad = float("inf")
    max_rad = 0.0
    total_sum = 0.0
    msums = [0] * len(centroids)
    for i, ctr in enumerate(centroids):
        for pt in clusters[i]:
            # print(pt.shape)
            # print(ctr.shape)
            msums[i] += np.linalg.norm(ctr - pt)  # D_vect(ctr, pt)
        total_sum += msums[i]
        # min_rad = min(min_rad, msums[i])
        # max_rad = max(max_rad, msums[i])
        print("{}\t{}".format("x", msums[i]))
