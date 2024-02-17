#!/usr/bin/python3
from clustering_driver import *
from clustering_imports import *
from tree_search_operations import *
from likelihood_helpers import *

def difference(m1, m2, noise=1):
  return jnp.sqrt(jnp.sum(((m1 - m2) / noise) ** 2))

def diff(m1,m2,noise):
  return float(
    jnp.sqrt(
        jnp.sum(((m1.flatten() - m2.astype(jnp.float32).ravel())/noise) ** 2)
    ).astype(jnp.float32)
  )

def psearch_leaf(T, idx, data_list, dbest, nearest_neighbor, noise,TAU=1):
  dist = diff(T.m1, data_list[idx].m1, noise)
  if dist < dbest[0]:
    dbest[0] = dist
    nearest_neighbor[0] = idx

def level_patient_search(
  node_list,
  data_list,
  input_list,
  TAU=0.4
):
  noise = calculate_noise(input_list)
  lambda_square = noise**2
  print(noise)
  likelihood_prime = [0.0 for _ in range(len(input_list))]
  likelihood_idx = [0 for _ in range(len(input_list))]
  nq = deque()
  start_time = time.perf_counter()
  sortkey = lambda x:-x[1] if x[1] != 0 else -1 * float('Inf')
  k = len(node_list[0].children)
  for i, T in enumerate(input_list):

    nq.append(0)
    nq.append(None)
    rq = deque()
    fb = node_list[0].cluster_radius/noise
    rq.append(fb)
    rq.append(None)
    # rq
    dbests = []
    nns = []
    skey = lambda x: node_list
    prev_index = 0
    node_index = 0
    cq = deque()
    # cq.append((0,fb))
    for c in node_list[0].children:
      cq.append((c, 0))
      cq.append((None,None))
      depth_counter = 1
      
      nn = [None]
      mdist = [float("Inf")]
      dbest = [float("Inf")]
      while len(cq):
        ncounter = 0
        while cq[0][0] != None:
          pref_index = node_index
          node_index,Rprev = cq.popleft()#nq.popleft()
          # print(node_index)
          Rprev = Rprev / noise
          # print(diff(T.m1, node_list[node_index].val.m1, 1),[((diff(T.m1, node_list[c].val.m1, 1)/node_list[c].cluster_radius),node_list[c].cluster_radius) for c in node_list[node_index].children],node_list[node_index].cluster_radius)
          # sys.exit()
          # Rprev = rq.popleft()
          
          # if node_list[node_index].data_refs != None:
          #   # R = node_list[node_index].cluster_radius/noise
          #   # C = diff(T.m1, node_list[node_index].val.m1, noise)
          #   # if C > R and C - R > dbest[0]:
          #   #   continue

          #   for index in node_list[node_index].data_refs:
          #     psearch_leaf(T, index, data_list, dbest, nn, noise, TAU)
          #   continue
          # if Rprev < dbest[0]:
          #   continue
          for c in node_list[node_index].children:
            if node_list[c].data_refs!=None:
              for index in node_list[c].data_refs:
                psearch_leaf(T, index, data_list, dbest, nn, noise, TAU)
              continue
            elif node_list[c].cluster_radius == 0:
              for index in node_list[c].children[0].data_refs:
                psearch_leaf(T, index, data_list, dbest, nn, noise, TAU)
              continue
            R = node_list[node_index].cluster_radius/noise
            C = diff(T.m1, node_list[c].val.m1, noise)
            if C > dbest[0]:
              continue
            new_bound = 0
            # if C > R + Rprev:
            #   continue
            # if C < Rprev:
            #   continue
            # if C > Rprev:
            #   continue
            if C > R:
              # new_bound = min(Rprev,2*R)*noise
              continue
              # new_bound = min(C-Rprev - R), 2 * R)*noise
            elif C < R:
              new_bound = (R - C) * noise
            # elif node_list[c].cluster_radius == 0:
            #   cq.append((c,0))
            #   continue
            if new_bound != 0:
              cq.append((c, new_bound))
              # nq.append(c)
              # rq.append(new_bound)

        # nq.popleft()
        # rq.popleft()
        cq.popleft()
        if len(cq):
          cq = deque(sorted(cq, key=sortkey)[0:max(k,int((k)**(depth_counter)))])
          depth_counter +=1
          # print(cq)
          cq.append((None,None))

          # sorted(nq)
          # nq.append(None)
          # rq.append(None)
      dbests.append((nn[0], dbest[0]))
    sortkey2 = lambda x: x[1]
    dbests = sorted(dbests, key=sortkey2)
    nns.append(dbests[0][0])
    # print(ncounter)
    likelihood_prime[i] = jnp.exp(
      -1.0
      * (difference_calculation(T.m1, data_list[nns[0]].m1, noise) ** 2)
      / (2 * lambda_square)
    )
    likelihood_idx[i] = nns[0]
    if not i % 20:
      print(i)
  likelihood_prime = postprocessing_adjust(likelihood_prime, noise, 1)
  end_time = time.perf_counter() - start_time
  LOGGER.info("{},{}".format(len(input_list) ** 2, end_time))
  return likelihood_prime, likelihood_idx