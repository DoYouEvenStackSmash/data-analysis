import numpy as np
from collections import deque

def weighted_sample(weights):
    """
    Sample a weighted probability distribution
    returns an index
    """
    total_w = weights / np.sum(weights)
    sample_val = np.random.uniform(0, 1)
    for idx, w in enumerate(total_w):
        sample_val -= w
        if sample_val <= 0:
            return idx
    return len(weights) - 1

M = np.random.randint(0, 500, (10000, 1, 2))

# initialize not chosen

def distance(k,m):
    """
    Distance function between k,m
    """
    return np.sqrt(np.sum(np.array([np.power(i, 2) for i in (k - m).flatten()])))
    # return D(k,m)

def assign_clusters(data, centroids):
    """
    Assign data to clusters using distance from nearest centroid
    """
    clusters = [[] for _ in centroids]
    for img in data:
        min_dist = float("inf")
        nn = None
        for idx, ctr in enumerate(centroids):
            if np.array_equal(img, ctr):
                continue
            pdist = distance(img, ctr)
            if pdist < min_dist:
                min_dist = pdist
                nn = idx
        if nn != None:
            clusters[nn].append(img)
    ret_clus = []
    for i in range(len(clusters)):
        if not len(clusters[i]):
            continue
        
        clusters[i] = np.stack([j for j in clusters[i]])

    return clusters

def update_centroids(clusters, old_centroids):
    """
    Recenter the old centroids to be the mean of the new clusters
    """
    centroids = [np.zeros((1, 2)) for _ in clusters]
    for idx, cluster in enumerate(clusters):
        rep = old_centroids[idx]
        if len(cluster) and len(cluster[0]):
            rep = np.mean(cluster, axis=0)
        centroids[idx] = rep
    return np.array(centroids)

def kmeanspp(M, k, centroids, not_chosen, chosen):
    """
    Compute a probably-better-than-random set of k centroids given an arr
    """
    for _ in range(k):
        weights = np.zeros(len(not_chosen))
        D = lambda ck, m: np.sqrt(
            np.sum(np.array([np.power(i, 2) for i in (ck - m).flatten()]))
        )
        for idx, mdx in enumerate(not_chosen):
            m = M[mdx]
            min_dist = float("inf")
            for ck in centroids:
                min_dist = min(min_dist, D(ck, m))
            weights[idx] = np.power(min_dist, 2)

        selected_point = weighted_sample(weights)
        centroids.append(M[not_chosen[selected_point]])

        chosen.add(not_chosen[selected_point])
        not_chosen.remove(not_chosen[selected_point])

    centroids = np.array(centroids)
    return centroids

def kmeans(M, k, max_iters=100):
    """
    Implementation of k means algorithm
    """
    start_center = np.random.randint(M.shape[0])

    centroids = [M[start_center]]

    not_chosen = deque()
    chosen = set()
    chosen.add(start_center)

    for i in range(len(M)):
        if i == start_center:
            continue
        not_chosen.append(i)

    # initialize centroids using kmeans++
    centroids = kmeanspp(M, k, centroids, not_chosen, chosen)
    # centroids = M[np.random.choice(1000, k, replace=False)]

    for _ in range(max_iters):
        clusters = assign_clusters(M, centroids)
        new_centroids = update_centroids(clusters, centroids)
        if np.array_equal(centroids,new_centroids):
            break

        centroids = new_centroids
    return clusters, centroids



k = 15
clusters, centroids = kmeans(M, k)
centers = []
reclus = []
# print(clusters)
for idx,k in enumerate(clusters):
    if not len(k):
        continue
    reclus.append(np.array(clusters[idx]))
    centers.append(centroids[idx])

centers = np.array(centers)
