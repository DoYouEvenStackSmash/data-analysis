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


# initialize M with M elements of shape (1,2)
M = np.random.randint(0, 100, (100, 1, 2))
print(M)
# initialize not chosen


def kmeans(data, k, max_iters):
    """
    Wrapper function for single level k means clustering
    """
    start_center = np.random.randint(data.shape[0])

    centroids = [data[start_center]]

    not_chosen = deque()
    chosen = deque()
    chosen.append(start_center)

    for i in range(len(data)):
        if i == start_center:
            continue
        not_chosen.append(i)

    centroids = kmeanspp(data, k, centroids, not_chosen, chosen)
    return centroids


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

        chosen.append(not_chosen[selected_point])
        not_chosen.remove(not_chosen[selected_point])

    centroids = np.array(centroids)
    return centroids
    # return centroids


k = 7
centroids = kmeans(M, k)

# print(weighted_sample(weights))
