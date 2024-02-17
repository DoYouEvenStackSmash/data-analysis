k = clusters 
c = cutoff
r = iterations
m = input
initialize root node id = 0
create a map for id:Node called NMap
create two queues, nq and dq for nodes and data respectively
add 0 to nq, add the set of M to queue dq

while nq not empty
    curr = nmap(nq.pop())
    dataset = dq.pop()
    if dataset.size > C: set node radius to appropriate distance from mean

    perform i within R iterations of kmeans on dataset
    centroids,labels = kmeans(dataset)
    create nnew (id:Node) in nmap for each centroid
    add each id to curr.children and to nq
    add each set of labels to dq
  
  else:
    perform i within R iterations of kmedoids on datasets
    see paper
    medoids, labels = kmedoids(dataset)

    for each mediod, labels:
      create a new id:node in nmap with cluster width
      add labels to node.data,add node's id to curr.children


return node map

min_dist = inf
pij = mat
for i in range(M)
  for j in range(N)
    p_ij = dist(M_i, N_j)

maxmat = []
for row in p_ij:
  maxmat(i) = argmin(row)

# maxmat contains 

def SearchLeaf(T,D,dbest,nn)
  if dist(T,D) < dbest:
    dbest = dist(T,D)
    nn = D

def SearchNode(T,N,irad,dbest,nn)
  if isleaf(N):
    for d in N.dref do
      searchLeaf(T,D,dbest,nn)
  elif irad < dbest:
    bound_arr <- [(N.radius - dist(T,C),C)] for c in N.children
    bound_arr.sort()
    for p in range(bound_arr)
      pivot,C = bound_arr[p]
      SearchNode(T,C,irad,dbest,nn)
      for s in [0 < i < len(bound_arr), i != p]:
        spivot, sC <- bound_arr[s]
        off_t = sum(bound_arr[0<j<len(bound_arr),j != p, j != s ])
        SearchNode(T, sC, irad - pivot + offt, dbest,nn)
      
