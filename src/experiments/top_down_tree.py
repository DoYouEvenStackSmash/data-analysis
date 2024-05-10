from sklearn.cluster import KMeans
import numpy as np
def get_data_size(data,k=2):
  d = 1
  while k**d < len(data):
    d+=1
  sd = k**d+k
  while k**d > 1:
    d-=1
    sd += k**d
  return sd

def get_data_terms(data,k=2):
  d = 1
  while k**(d-1) < len(data):
    d+=1
  kterms = np.arange(d)
  terms = [0] * d
  a = 1
  for i in range(d):
    terms[i] = np.sum(a * k ** kterms[0:i])
  return terms

from collections import deque
def tree_build(N ,k=2):

  data = np.random.randn(N,2)
  print(data.shape)
  o_data_indices = np.arange(len(data))
  kmeans = KMeans(n_clusters=k)
  
  # print(data[data_indices])
  labels = kmeans.fit_predict(data[o_data_indices])
  associated_data = [[] for _ in range(k)]
  for i,elem in enumerate(labels):
    associated_data[elem].append(o_data_indices[i])
  new_data_q = deque()
  base_q = deque()
  for i in range(k):
    base_q.append(1)
    new_data_q.append(i)
  base_q.append(None)
  new_data_q.append(None)
    
  data_q = deque()
  for i,elem in enumerate(associated_data):
    data_q.append(associated_data[i])
  data_q.append(None)
  
  global_data = [None]*(get_data_size(data, k)+1)
  tree_idx = [None] * (get_data_size(data, k)+1)
  for i,d in enumerate(data):
    global_data[i] = data[i]
  # print(len(global_data))
  base_offt = len(data)
  for i in range(len(kmeans.cluster_centers_)):
    global_data[base_offt+i] = kmeans.cluster_centers_[i]
  print(len(global_data), len(tree_idx))
  # tree_idx[len(data):len(data)+k] = [i for i in range(k)]
  base_addr_terms = get_data_terms(data, k)
  
  depth = 1
  print(base_offt)
  while len(new_data_q):
    while new_data_q[0] != None:
      
      data_indices = data_q.popleft()
      # base_addr = base_q.popleft()
      base_q.popleft()
      
      local_addr = new_data_q.popleft()
      if depth >= len(base_addr_terms):
        continue
      base_addr = base_offt + base_addr_terms[depth]
      offt = k**(depth-1) if depth != 0 else 1

      win_start = base_addr + offt*local_addr
      
      win_end = win_start + k

      next_base = (local_addr+1) * k**(depth+1) + base_addr

      if len(data_indices) <= k and len(data_indices) > 0:

        for i in range(len(data_indices)):

          tree_idx[win_start + i] = -1 * (data_indices[i] + 1)
        continue
      
      kmeans = KMeans(n_clusters=k,init='k-means++')
      labels = kmeans.fit_predict(data[data_indices])
      # centroids = kmeans.cluster_centers_
      
      # labels = kmeans.fit_predict(data[data_indices])
      global_data[win_start:win_end] = kmeans.cluster_centers_
      # tree_idx[win_start:win_end] = [i for i in range(k)]
      associated_data = [[] for _ in range(k)]

      for i,elem in enumerate(labels):
        associated_data[elem].append(data_indices[i])
      for i,elem in enumerate(associated_data):
          tree_idx[win_start + i] = i
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
  return tree_idx,global_data,base_addr_terms
  
def level_order_traversal(tree_idx,global_data, base_offt,k,terms):
  new_data_q = deque()
  base_q = deque()
  
  for i in range(k):
    base_q.append(1)
    new_data_q.append(i)
  base_q.append(None)
  new_data_q.append(None)
  depth = 1
  while len(new_data_q):
    while new_data_q[0] != None:
      
      # base_addr = base_q.popleft()
      base_q.popleft()
      
      local_addr = new_data_q.popleft()
      if depth >= len(terms):
        continue
      base_addr = base_offt + terms[depth]
      win_start = base_addr + k**(depth-1)*local_addr
      next_base = (local_addr+1) * k**(depth+1) + base_addr if depth != 0 else 1

      win_end = win_start + k
      print(win_start)
      for i,e in enumerate(tree_idx[win_start:win_end]):
        if e == None:
          continue
        if e < 0:
          tree_idx[-1 * (e + 1)] = "DATA" + f'{win_start+i}'
        # else:
        #   # new_data_q.extend([i])
        #   # base_q.extend([next_base])
        # else:
        #   # tree_idx[win_start+i] = f'{win_start+i}->{base_offt + next_base + k*i};'
        #   # for x in range(1,k):
        #     # tree_idx[win_start+i] = tree_idx[win_start+i] + f'{win_start+i}->{base_offt + next_base + k*i+x};'
        new_data_q.extend([i])
        base_q.extend([next_base])
    depth+=1
    new_data_q.popleft()
    base_q.popleft()
    if len(new_data_q):
      new_data_q.append(None)
      base_q.append(None)
    
def main():
  tb,gd,terms = tree_build(9,2)
  print(len(tb))
  print(terms)
  level_order_traversal(tb, gd,9, 2,terms)
  print(tb)
  # print(global_data)
if __name__ == '__main__':
  main()
      
      