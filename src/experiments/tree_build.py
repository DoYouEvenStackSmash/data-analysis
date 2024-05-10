#!/usr/bin/python3
from collections import deque
def tree_build(k):
  # k = k 
  level_stack = deque()
  level_stack.extend([2 for _ in range(k)])
  parent_stack = deque()
  parent_stack.extend([i for i in range(k)])
  data_idx = lambda level_idx, k, idx: (level_idx-1) * k + idx
  data_list = [f'0-0',f'0-1']
  print(parent_stack)
  stop_condition = lambda d: d > 4
  # leaves = [[] for _ in range()
  levels = [[0,1]]
  while len(level_stack):
    level_idx = level_stack.popleft()
    parent_idx = parent_stack.popleft()
    
    if stop_condition(level_idx):
      continue
    if len(levels) < level_idx:
      levels.append([])
    # levels[level_idx].extend
    # data_hold = level_idx * k + idx
    # if labels < k:
      
    hold = parent_idx
    # data_labels[level * hold * k + i ]
    levels[level_idx-1].extend([hold*k + i for i in range(k)])
    parent_stack.extend([hold*k + i for i in range(k)])
    level_stack.extend([level_idx + 1 for _ in range(k)])
    data_list.extend([f"{level_idx-1}-{hold*k + i}" for i in range(k)])
    # data_list.append([])
  return levels, data_list

l,d = tree_build(2)
for level in l:
  print(level)
print(d)