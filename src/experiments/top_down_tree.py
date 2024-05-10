from sklearn.cluster import KMeans
import numpy as np
def get_data_size(data,k=2):
  d = 1
  while k**d < len(data):
    d+=1
  sd = k**d
  while k**d > 1:
    d-=1
    sd += k**d
  return sd

from collections import deque
def main():
  k=2
  data = np.random.randn(15,2)
  print(data.shape)
  data_indices = np.arange(len(data))
  kmeans = KMeans(n_clusters=k)
  # print(data[data_indices])
  labels = kmeans.fit_predict(data[data_indices])
  associated_data = [[] for _ in range(k)]
  for i,elem in enumerate(labels):
    associated_data[elem].append(data_indices[i])
  new_data_q = deque()
  base_q = deque()
  for i in range(k):
    base_q.append(1)
    new_data_q.append(i)
  base_q.append(None)
  new_data_q.append(None)
    
  data_q = deque()
  for elem in associated_data:
    data_q.append(elem)
  data_q.append(None)
  
  global_data = [None]*get_data_size(data, k)
  tree_idx = [None] * len(global_data)
  global_data[0:len(data)] = data
  print(len(global_data))
  global_data[len(data):len(data)+k] = kmeans.cluster_centers_
  tree_idx[len(data):len(data)+k] = [i for i in range(k)]
  
  base_offt = len(data)
  depth = 1
  
  while len(new_data_q):
    while new_data_q[0] != None:
      data_indices = data_q.popleft()
      base_addr = base_q.popleft()
      local_addr = new_data_q.popleft()
      
      win_start = base_offt + base_addr + k*local_addr
      
      win_end = win_start + k

      next_base = (local_addr+1) * k**(depth+1) + base_addr if depth != 0 else 1

      if len(data_indices) <= k:
        tree_idx[win_start:win_start+len(data_indices)] = [-1 * i for i in data_indices]
        continue
      
      kmeans = KMeans(n_clusters=k,init='k-means++')
      labels = kmeans.fit_predict(data[data_indices])
      # centroids = kmeans.cluster_centers_
      
      # labels = kmeans.fit_predict(data[data_indices])
      global_data[win_start:win_end] = kmeans.cluster_centers_
      tree_idx[win_start:win_end] = [i for i in range(k)]
      associated_data = [[] for _ in range(k)]

      for i,elem in enumerate(labels):
        associated_data[elem].append(data_indices[i])
      for i,elem in enumerate(associated_data):
        new_data_q.extend([i])
        base_q.extend([next_base])
        data_q.append(elem)
    depth+=1
    new_data_q.popleft()
    base_q.popleft()
    data_q.popleft()
    if len(new_data_q):
      new_data_q.append(None)
      base_q.append(None)
      data_q.append(None)
  print(tree_idx)
  
  # print(global_data)
if __name__ == '__main__':
  main()
      
      