import torch


def calculate_distance(T, S):
    return torch.sqrt(torch.sum((T - S) ** 2, dim=1))


a = torch.rand(10, 4)
b = torch.rand(10, 4)
print(calculate_distance(a, b))
