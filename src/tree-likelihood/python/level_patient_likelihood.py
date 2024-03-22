#!/usr/bin/python3

from clustering_driver import *
from clustering_imports import *
from tree_search_operations import *
from likelihood_helpers import *

def search_leaf(node, T, data_list, dbest, nearest_neighbor_idx, noise=1):
  for omega_idx in node.data_refs:
    # apply experimental filter to omega
    omega_phi = conv_jax_apply_d1m2_to_d2m1(T, data_list[omega_idx])
    # calculate distance with noise
    d = difference(T.m1, omega_phi, noise)
    if d < dbest[0]:
      dbest[0] = d
      nearest_neighbor_idx[0] = omega_idx

def is_leaf(node):
  return (node.data_refs != None or node.cluster_radius == 0)

def _level_patient_likelihood(node_list, data_list, T, noise=1):
  """single target search

  Args:
      node_list (_type_): _description_
      data_list (_type_): _description_
      T (_type_): _description_
      noise (int, optional): _description_. Defaults to 1.
  """
  
  # set up placeholders
  closest_distances = [float("Inf")]
  nearest_neighbors_idx = [-1]
  
  node_q = deque()
  depth_q = deque()
  #node_list[=]lambda c: node_list[c]
  # is_leaf = lambda node: node.data_refs != None or node.cluster_radius == 0
  for c in node_list[0].children:
    
    # insert children
    if node_list[c].children != None:
      node_q.append((c, 0))
      depth_q.append(1)
    
    # insert end flags
    node_q.append((None, None))
    depth_q.append((-1))
    
    dbest = [float("Inf")]
    nearest_neighbor_idx = [-1]
    
    # traverse everything
    while len(node_q):
      # traverse a level
      while node_q[0][0] != None:
        nidx,_ = node_q.popleft()
        curr_depth = depth_q.popleft()
        for c in node_list[nidx].children:
          # search leaf if needed
          if is_leaf(node_list[c]):
            search_leaf(node_list[c], T, data_list, dbest, nearest_neighbor_idx, noise)
            continue
          # check ctf bound
          in_bound_flag, _ = check_ctf_bound(node_list[c], node_list[nidx], T, noise)
          if in_bound_flag:
            node_q.append((c,0))
            depth_q.append(curr_depth + 1)
          else:
            pass
      # get end-of-level flag
      node_q.popleft()
      depth_q.popleft()
      
      if len(node_q):
        #preprocess next level here
        # insert end flags
        node_q.append((None, None))
        depth_q.append(-1)
    if nearest_neighbor_idx[0] != -1:
      if dbest[0] < closest_distances[0]:
        nearest_neighbors_idx[0] = nearest_neighbor_idx[0]
        closest_distances[0] = dbest[0]
    
        
  return nearest_neighbors_idx[0]


def level_patient_likelihood(node_list, data_list, input_list):
  """single target likelihood evaluation via breadth first search with a bound
  May not be backwards compatible

  Args:
      node_list (_type_): _description_
      data_list (_type_): _description_
      input_list (_type_): _description_
  """
  
  likelihoods = [0.0 for _ in range(len(input_list))]
  likelihood_idx = [0 for _ in range(len(input_list))] 
  
  # calculate noise
  noise = calculate_noise(input_list)
  lambda_square = noise**2
  
  start_time = time.perf_counter()
  out_of_bound_counter = 0
  for i,y in enumerate(input_list):
    best_match_idx = _level_patient_likelihood(node_list, data_list, y,noise)
    
    if best_match_idx < 0:
      out_of_bound_counter += 1
      continue
    xi = data_list[best_match_idx]
    num = difference(y.m1, conv_jax_apply_d1m2_to_d2m1(y, xi), noise)
    likelihoods[i] = jnp.exp(-1 * num / (2 * lambda_square))
    likelihood_idx[i] = best_match_idx
    if not i % 20:
      print(i)
  likelihoods = postprocessing_adjust(likelihoods, noise, 1)
  end_time = time.perf_counter() - start_time
  LOGGER.info("{},{}".format(len(input_list) ** 2, end_time))
  return likelihoods, likelihood_idx
    