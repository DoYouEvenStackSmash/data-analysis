import numpy as np
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
data = [0]*30
terms = get_data_terms(data,3)

padded_data = [0] * 30
padded_data.extend([None]*(terms[-1] - len(padded_data)))

print(padded_data)
idx = lambda kt,level,k,local_addr: kt[level] + k**level * (local_addr)
print(terms)
local_idx_q = deque([0,1,2,None])
base_q = deque([0,0,0,None])
# curr_depth = 1

# print(len(padded_data))
k=3
while len(local_idx_q):
  while local_idx_q[0] != None:
    local_addr = local_idx_q.popleft()
    parent_idx = base_q.popleft()
    # print(local_addr, parent_addr)
    base_addr = terms[parent_idx]
    # offt = int((terms[curr_depth-1] - 1) / k)
    curr_addr = base_addr + (local_addr)*k

    
    win_start = curr_addr
    
    win_end = min(win_start + k, len(padded_data))
    if win_start > len(padded_data):
      # print(len(padded_data))
      continue
    print(win_start,end=',')

    for i,elem in enumerate(padded_data[win_start:win_end]):
      if elem == None:
        continue
      # print(win_start,win_end)
      padded_data[win_start+i] =curr_addr+i if padded_data[win_start+i] == 0 else -1
      # print()
      val = (i%k + i)*k
      local_idx_q.extend([i+base_addr*parent_idx])
      base_q.extend([(parent_idx+1)])
  # curr_depth += 1

  
  local_idx_q.popleft()
  base_q.popleft()
  if len(local_idx_q):
    local_idx_q.append(None)
    base_q.append(None)
print(terms)
print(padded_data)