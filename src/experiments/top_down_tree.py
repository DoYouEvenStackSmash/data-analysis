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
  new_data_q = deque([i for i in range(k)])
  base_q = deque([1 for _ in range(k)])
  new_data_q.append(None)
  base_q.append(None)
  # for i in range(k):
  #   base_q.append(1)
  #   new_data_q.append(i)
  # base_q.append(None)
  # new_data_q.append(None)
    
  data_q = deque()
  for i,elem in enumerate(associated_data):
    data_q.append(associated_data[i])
  data_q.append(None)
  kt = get_data_terms(data, k)
  row_lens = [(kt[i+1] - kt[i]) for i in range(len(kt)-1)]
  global_data = [None]* (len(data) + k**(len(kt)+1))
  tree_idx = [None] * (len(data) + k**(len(kt)+1))
  for i,d in enumerate(data):
    global_data[i] = data[i]
  # print(len(global_data))
  base_offt = len(data)
  for i in range(len(kmeans.cluster_centers_)):
    global_data[base_offt+i] = kmeans.cluster_centers_[i]
  # print(len(global_data), len(tree_idx))
  # tree_idx[len(data):len(data)+k] = [i for i in range(k)]
  # base_addr_terms = get_data_terms(data, k)
  
  # depth = 1
  # print(base_offt)
  while len(new_data_q):
    while new_data_q[0] != None:
      
      data_indices = data_q.popleft()
      
      parent_idx = new_data_q.popleft()
      
      curr_level = base_q.popleft()
      
      wbin = (parent_idx) * k + 1
      base_addr = 0
      if curr_level >= len(row_lens):
        base_addr = sum(row_lens[0:curr_level-1])
      else:
        base_addr = row_lens[curr_level]
      wstart = base_offt + base_addr + wbin
      wend = wstart + k
      
      if len(data_indices) <= k and len(data_indices) > 0:
        for i in range(len(data_indices)):
          tree_idx[wstart + i] = -1 * (data_indices[i] + 1)
        continue
      
      kmeans = KMeans(n_clusters=k,init='k-means++')
      labels = kmeans.fit_predict(data[data_indices])
      # centroids = kmeans.cluster_centers_
      
      # labels = kmeans.fit_predict(data[data_indices])
      global_data[wstart:wend+1] = kmeans.cluster_centers_
      # tree_idx[wstart:win_end] = [i for i in range(k)]
      associated_data = [[] for _ in range(k)]

      for i,elem in enumerate(labels):
        associated_data[elem].append(data_indices[i])
      for i,elem in enumerate(associated_data):
          print(wstart)
          tree_idx[wstart + i] = i
          
          # if curr_level + 1 < len(row_lens):
          new_data_q.extend([wbin+i])
          base_q.extend([curr_level+1])
          # print(elem)
          data_q.extend([associated_data[i]])

    new_data_q.popleft()
    base_q.popleft()
    data_q.popleft()
    if len(new_data_q):
      new_data_q.append(None)
      base_q.append(None)
      data_q.append(None)
  print(tree_idx)
  tree_idx[base_offt] = 0
  for i in range(k):
    # if tree_idx[base_offt + i + 1] == 0:
    tree_idx[base_offt + i + 1] = i
    
  return tree_idx,global_data,kt,row_lens
  
def level_order_traversal(tree_idx,global_data, base_offt,k,kt,row_lens):

  
  new_data_q = deque([i for i in range(k)])
  new_data_q.append(None)
  base_q = deque([1 for _ in range(k)])
  base_q.append(None)
  depth = 1

  
  # print(base_str)
  while len(new_data_q):
    while new_data_q[0] != None:
      
      # data_indices = data_q.popleft()
      
      parent_idx = new_data_q.popleft()
      
      curr_level = base_q.popleft()
      
      wbin = (parent_idx) * k + 1
      base_addr = 0
      if curr_level >= len(row_lens):
        base_addr = sum(row_lens[0:curr_level-1])
      else:
        base_addr = row_lens[curr_level]
      wstart = base_offt + base_addr + wbin
      wend = wstart + k
        # print(wstart)
      # win_end = min(len(glowstart + k
      # print(wstart)
      for i,e in enumerate(tree_idx[wstart:wend]):
        if e == None:
          continue
        if e < 0:
          
          tree_idx[-1 * (e + 1)] = 1 if tree_idx[-1 * (e + 1)] == None else tree_idx[-1 * (e + 1)] + 1
          # print(f'{wstart+i}->{tree_idx[-1*(e+1)]};')
        else:
        #   # new_data_q.extend([i])
        #   # base_q.extend([next_base])
        # else:
          print(f'{wstart+i}->{base_offt + (row_lens[curr_level+1] if curr_level + 1 < len(row_lens) else sum(row_lens[0:curr_level])) + (wbin+i)*k+1};')
          # for x in range(k):
          #   addr = base_offt + (row_lens[curr_level+1] if curr_level + 1 < len(row_lens) else sum(row_lens[0:curr_level])) + (wbin+i)*k+1 + x
          #   if addr > len(global_data):
          #     continue
          #   if tree_idx[addr] == None:
          #     continue
          #   tree_idx_str = tree_idx_str + f'{wstart+i}->{addr};'
          # print(tree_idx_str)
        # if curr_level + 1 < len(row_lens):
          new_data_q.extend([wbin+i])
          base_q.extend([curr_level+1])
    # depth+=1
    new_data_q.popleft()
    base_q.popleft()
    if len(new_data_q):
      new_data_q.append(None)
      base_q.append(None)
    
def main():
  tb,gd,kt,row_lens = tree_build(10,2)
  print(len(tb))
  # print(terms)
  level_order_traversal(tb, gd,10, 2,kt,row_lens)
  print(tb)
  # print(global_data)
if __name__ == '__main__':
  main()
      
      