
# preallocate 
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
  k = 4
  data = np.random.randn(13)
  data_len = get_data_size(data,k)
  data = list(data)
  data.extend([None] * (data_len - len(data)+1))
  data_idx = np.arange(len(data)) * 0
  
  new_data_q = deque([0,1,2,3,None])
  base_q = deque([1,1,1,1,None])

  last_depth = 0
  depth = 1
  base_offt = 4
  data_idx[0:base_offt] = -1
  while len(new_data_q):

    while new_data_q[0] != None:
      # print(base_q[0],end=":")
      base_addr = base_q.popleft()
      local_addr = new_data_q.popleft()
      
      win_start = base_offt + base_addr + k*local_addr
      
      win_end = win_start + k

      print(data_idx[win_start:win_end],end='-')
      data_idx[win_start:win_end] = win_start
      
      next_base = (local_addr+1) * k**(depth+1) + base_addr if depth != 0 else 1
      
      for i,elem in enumerate(data[win_start:win_end]):
        if elem != None:
          new_data_q.extend([i])
          base_q.extend([next_base])
        else:
          data_idx[win_start+i] = -2
          #data_q.extend([associated_data])
    print(base_q[-1])
    depth+=1
    new_data_q.popleft()
    base_q.popleft()
    if len(new_data_q):
      new_data_q.append(None)
      base_q.append(None)
  for i in range(0,len(data_idx),k):
    print(data_idx[i:i+k])
  


if __name__ == '__main__':
  main()