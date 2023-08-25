import sys
sys.path.append("..")

from cluster_main import *


def assign_clusters(dlen, mlist, distances):
    clusters = [[] for _ in mlist]
    
    for idx in range(dlen):
        distances_to_medioids = distances[idx, mlist]
        nearest_medioid_index = np.argmin(distances_to_medioids)
        clusters[nearest_medioid_index].append(idx)
    
    return clusters

def calculate_sum(clusters, mlist, distances):
    total_sum = np.sum([distances[idx, midx] for midx in mlist for idx in clusters[mlist.index(midx)]])
    return total_sum

def update_medioids(clusters, mlist, distances):
    new_mlist = []
    
    for midx in mlist:
        cluster = clusters[mlist.index(midx)]
        cluster_distances = np.sum(distances[cluster][:, cluster], axis=1)
        new_mlist.append(cluster[np.argmin(cluster_distances)])
    
    return new_mlist

def preprocess(M,k=3):

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
    return mlist,distances


def kmedioids(M, k=5, max_iter=100):

    n = M.shape[0]
    mlist, distances = preprocess(M, k)

    # Precompute distances
    # distances = np.sqrt(np.sum(np.square(M[:, np.newaxis, :] - M), axis=2))
    
    total_sum = float('inf')
    for _ in range(max_iter):
        clusters = assign_clusters(n, mlist, distances)
        mlist = update_medioids(clusters, mlist, distances)
        new_sum = calculate_sum(clusters, mlist, distances)
        if new_sum == total_sum:
            break
        total_sum = new_sum
    
    return clusters, mlist

# # Call kmedioids function
# clusters, mlist = kmedioids(M, k, max_iter=100)
# print(set(mlist))
