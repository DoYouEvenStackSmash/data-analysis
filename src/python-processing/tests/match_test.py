#!/usr/bin/python3
import sys
sys.path.append("../")
from clustering_imports import *
from clustering_driver import *

k_arr = [3]
R_arr = [30]
C_arr = [35]
D_arr = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,]# 8172]
N_arr = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,]# 8172]
base = np.random.randint(0, 2000,(4096,4096,4096))
for k in k_arr:
  for R in R_arr:
    for C in C_arr:
      for N in N_arr:
        for D in D_arr:
          M = base[:N,:D,:D]
          node_list, data_list = hierarchify(M, k,R, C)
          np.random.shuffle(M)
          tree_match_indices, tree_match_distances = search_tree_associations(node_list, data_list, M)
