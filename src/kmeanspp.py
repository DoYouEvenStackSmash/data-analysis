import numpy as np
from collections import deque

M = np.random.randint(0,100,(100,1,2))

print(M)
start_center = np.random.randint(M.shape[0])
centers = [M[start_center]]
nc = deque()
for i in range(len(M)):
  if i == start_center:
    continue
  nc.append(i)


weights = np.zeros(len(nc))
D = lambda k,m: np.sqrt(np.sum(np.array([np.power(i,2) for i in (k - m).flatten()])))
for idx, mdx in enumerate(nc):
  m = M[mdx]
  min_dist = float('inf')
  for k in centers:
    min_dist = min(min_dist, D(k,m))
  weights[idx] = np.power(min_dist,2)


def weighted_sample(weights):
  total_w = weights / np.sum(weights)
  sample_val = np.random.uniform(0,1)
  for idx, w in enumerate(total_w):
    sample_val -= w
    if sample_val <= 0:
      return idx
  return len(weights) - 1

print(weighted_sample(weights))

selected_point = weighted_sample(weights)
centers.append(M[nc[selected_point]])
nc.remove(nc[selected_point])
