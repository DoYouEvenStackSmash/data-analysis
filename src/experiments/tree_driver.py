from torch_tree_build import *
RESHAPE = False

def document(tree_dictionary, global_data, radius_data, N, k,filename='tree_stats.txt'):
    with open(filename, 'w') as f:
      f.write(f"N={N}, k={k}\n")
      f.write(f'+{"="*50}+\n')
      for i, e in enumerate(global_data):
          if i == N:
              f.write(f'+{"="*50}+\n')

          tree_idx_str = str(tree_dictionary[i]) if i in tree_dictionary else "None"
          # radius_data_str = str(radius_data[i].item()) if radius_data[i] is not None else "None"
          radius_data_str = "RAD" if radius_data[i] is not None else "None"
          # global_data_str = str(global_data[i].numpy()) if global_data[i] is not None else "None"
          global_data_str = "DATA" if global_data[i] is not None else "None"
          # if i < N:
          #     global_data_str = "DATA"

          
          f.write(f"| {i}:\t{tree_idx_str}\t{radius_data_str}\t{global_data_str}\n")

def main():
    N = 2000
    D = 128
    D2 = 128
    k = 5
    data_type=torch.float64
    simulated,experimental,ctfs = None,None,None
    
    
    if RESHAPE:
      simulated = torch.randn(N,D*D2,dtype=torch.complex64)#torch.tensor([i.ravel() for i in np.random.rand(N, 16384)])
      experimental = torch.randn(N,D*D2,dtype=torch.complex64)#torch.tensor([i.ravel() for i in np.random.rand(N, 16384)])
      # ctfs = torch.ones(N,D*D2,dtype=torch.complex64)
      experimental = c2r(experimental).view(experimental.shape[0], -1)
      simulated = c2r(simulated).view(simulated.shape[0], -1)
      # noises = c2r(noises).view(noises.shape[0], -1)
      ctfs = c2r(ctfs).view(ctfs.shape[0], -1)
    else:
      simulated = torch.randn(N,D*D2,dtype=data_type)#torch.tensor([i.ravel() for i in np.random.rand(N, 16384)])
      experimental = torch.randn(N,D*D2,dtype=data_type)#torch.tensor([i.ravel() for i in np.random.rand(N, 16384)])
      ctfs = torch.ones(N,D*D2,dtype=data_type)
    print(ctfs.shape)
    #d = data#torch.tensor([i.ravel() for i in np.random.rand(1, 2)])*10
    # ctfs = torch.tensor([i.ravel() for i in np.ones((N,D))])
    
    tree_idx, global_data,radius_data, kt, row_lens = scatter_tree_build(simulated, k)
    print(f"Tree build complete. \n\t{len(tree_idx)}")
    tree_dictionary = gather_tree_dictionary(tree_idx, global_data, N, k, kt, row_lens)
    document(tree_dictionary, global_data,radius_data, N, k)
    tree_dictionary, global_data, radius_data = tree_minimize(simulated, tree_dictionary,tree_idx, global_data, radius_data, N,k)
    document(tree_dictionary, global_data,radius_data, N, k,'post_compression.txt')
    print(f"Valid_CHECK: {sum(tree_idx[0:N])}")
    d1 = check_count(global_data[N:])
    d2 = check_count(global_data[N:])
    treesum = len(tree_dictionary) + sum(
        [len([i for i in v if i > N]) for k, v in tree_dictionary.items()]
    )
    print(f"TREE NODES: {treesum}")
    print(f"TREE REF ARRAY: {d1}/{len(global_data[N:])}  @ {d1/len(global_data[N:])} density")
    print(
        f"TREE DATA ARRAY:{d2}/{len(global_data[N:])} @ {d2/len(global_data[N:])} density"
    )
    # clusters = gather_clustered_data(tree_dictionary, global_data, N, k)
    # print(clusters)
    # draw_cluster_points(clusters)
    # with open("tree_stats.txt", 'w') as f:
    #     f.write(f"N={N}, k={k}\n")
    #     f.write(f'+{"="*50}+\n')
    #     for i, e in enumerate(global_data):
    #         if i == N:
    #             f.write(f'+{"="*50}+\n')

    #         tree_idx_str = str(tree_dictionary[i]) if i in tree_dictionary else "None"
    #         radius_data_str = str(radius_data[i].item()) if radius_data[i] is not None else "None"
    #         global_data_str = str(global_data[i].numpy()) if global_data[i] is not None else "None"
    #         # if i < N:
    #         #     global_data_str = "DATA"

            
    #         f.write(f"| {i}:\t{tree_idx_str}\t{radius_data_str}\t{global_data_str}\n")
    # # print("prepping...")
    # print(tree_dictionary)
    # exit()
    
    repl_tensor(global_data,1)
    gd = torch.stack(global_data)
    rd = torch.stack([i if i != None else torch.tensor(float('nan')) for i in radius_data ])
    # #v1
    if RESHAPE:
      pass
      # gd = r2c(gd,D,D2)
      # # rd = rd.
      # # print("starting search")
      # experimental = r2c(experimental, D, D2)
      # ctfs = r2c(ctfs,D,D2)
      # simulated = r2c(simulated,D,D2)
    for i,d in enumerate(simulated):
        # compact_level_order_search(d,tree_dictionary, rd, r2c(gd),ctfs[i], N, k, 0.1))
        print(f'{i}:{len(compact_level_order_search(d,tree_dictionary, rd, gd,ctfs[i], N, k, 0.0111))}')

    # print(tree_dictionary)
    # level_order_traversal(tree_dictionary, tree_idx,global_data, N,k)


if __name__ == "__main__":
    main()
