
# preallocate 
import numpy as np
# def get_data_size(data,k=2):
#   d = 1
#   # sd = 1
#   while k**d < len(elems):
#     d+=1
#     # print(d)
#   # sd += k**d
#   # print(d)
#   sd = k**d
#   # print(elems.shape)
#   while k**d > 1:
#     d-=1
#     sd += k**d
#   return sd

# #base_idx + local_idx * k ** level
# def tree_build(data, data_sz, k):
#   local_idx_q = deque()
#   curr_level_q = deque()
#   global_data = [None] * data_sz

#   while len(curr_level_q):
#     while curr_level_q[0] != None:
#       local_idx = local_idx_q.popleft()
#       base_idx = base_idx_q.popleft()
#       data_indices = data_q.popleft()
#       curr_level = curr_level_q.popleft()
#       kmeans = KMeans(n_clusters=k)
#       labels = kmeans.fit_predict(data[data_indices])
#       centroids = kmeans.cluster_centers_
#       window_start = k**(curr_level+1) * local_idx + base_idx + k**curr_level
#       window_end = window_start + k
#       global_data[window_start:window_end] = centroids
#       if len(labels) < k:
#         data_window_start = k**(curr_level+1) + local_idx

    
def get_data_size(data,k=2):
  d = 1
  # sd = 1
  while k**d < len(data):
    d+=1
    # print(d)
  # sd += k**d
  # print(d)
  sd = k**d
  # print(elems.shape)
  while k**d > 1:
    d-=1
    sd += k**d
  return sd
    
#     kmeans.fit_predict()
get_child_start = lambda k,curr_level,local_idx,base_idx: k**(curr_level) * local_idx + base_idx
get_child_end = lambda k,curr_level,local_idx,base_idx: get_child_start(k,curr_level,local_idx+1,base_idx)
from collections import deque
def main():
  
  k = 3
  data = np.random.randn(25)
  data_len = get_data_size(data,k)
  data = list(data)
  data.extend([None] * (data_len - len(data)+1))
  data_idx = np.arange(len(data))
  new_data_q = deque([0,1,2,None])
  base_q = deque([1,1,1,None])
  print(data_idx)
  last_depth = 0
  depth = 1
  while len(new_data_q):
    # print(base_q[0])
    while new_data_q[0] != None:
      print(base_q[0],end=":")
      base_addr = base_q.popleft()
      local_addr = new_data_q.popleft()
      win_start = base_addr + k*local_addr
      win_end = win_start + k
      print(win_start)

      
      print(data_idx[win_start:win_end])
      
      next_base = 0
      if depth != 0:
        next_base = (local_addr+1) * k**(depth+1) + base_addr
      else:
        next_base = 1
      for i,elem in enumerate(data[win_start:win_end]):
        if elem != None:
          new_data_q.extend([i])
          base_q.extend([next_base])
    
    # print(base_q[-1])
    print("-")
    
    depth+=1
    # print("end")
    new_data_q.popleft()
    base_q.popleft()
    if len(new_data_q):
      new_data_q.append(None)
      base_q.append(None)
  
  


if __name__ == '__main__':
  main()