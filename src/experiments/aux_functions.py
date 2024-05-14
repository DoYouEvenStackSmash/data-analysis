import torch

r2c = lambda x,d,d2: x.view(torch.cfloat).view(-1, d, d2)

c2r = lambda x: (
    torch.stack((x.real, x.imag), dim=-1)
    if x.dtype == torch.complex64
    else torch.stack((x.real, 0), dim=-1)
)

def calculate_radius(T, S, noise):
    # calculate distance between T and S
    return torch.sqrt(torch.sum(((T - (S)) ** 2),dim=1))

def calculate_distance(T, S,ctf, noise):
    # calculate distance between T and S*ctf
    return torch.sqrt(torch.sum(((T - (S*ctf)) ** 2),dim=1))

def check_count(arr):
    # density check function
    count = 0
    for i in range(len(arr)):
        if type(arr[i]) == type(None):
            continue
        count += 1
    return count

def repl_tensor(arr,flag):
    # replacement for None. 
    # lazy hack.
    count = 0
    for i in range(len(arr)):
        if type(arr[i]) == type(None):
            arr[i] = torch.empty(arr[0].shape)
            continue
        count += 1
    return count
  
