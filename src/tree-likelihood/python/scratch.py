
curr = node_queue.pop()
data = data_queue.pop()
curr.children = []
while len(q):
  if len(data) > C:
    centroids, clusters = kmeans(data)
    # create new nodes with centroids, add to curr
    # add data labels to queue
    for ctr in centroids:
      #create new node N
      N.id = len(node_list)
      N.val = ctr
      node_list.append(N)
      curr.children.append(N.id)
      node_queue.append(N)
      #add N.id to curr.children
      #add N to node_queue
    
    for cls in clusters:
      data_queue.append(cls)
      add cls to data_queue
    else:
      medoids, clusters = kmedoids(data)
      for mtr in medoid:
        #create new node N
        # N.val = mtr
        # N.data = cluster
        
        #add node to curr.children
        #add labels to 
      
      
    
  