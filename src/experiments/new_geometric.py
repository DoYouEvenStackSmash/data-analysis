import numpy as np


def get_data_terms(data, k=2):
    d = 1
    while k ** (d - 1) < len(data):
        d += 1
    kterms = np.arange(d)
    terms = [0] * d
    a = 1
    for i in range(d):
        terms[i] = np.sum(a * k ** kterms[0:i])
    return terms


from collections import deque

data = np.random.rand(256)
print(len(data))
k = 4
kt = get_data_terms(data, k)
print(kt)
padded_data = [0] * len(data)
padded_data.extend([None] * (kt[-1]))
# kt = get_data_terms(padded_data,k)
row_lens = [(kt[i + 1] - kt[i]) for i in range(len(kt) - 1)]

parent_idx_q = deque()
parent_idx_q.extend([i for i in range(k)])
# parent_idx_q[1] = 0
parent_idx_q.append(None)
parent_level_q = deque()
parent_level_q.extend([1 for _ in range(k)])
# parent_level_q[0] = 0
parent_level_q.append(None)

print(row_lens)
print(padded_data)
while len(parent_idx_q):
    while parent_idx_q[0] != None:
        parent_idx = parent_idx_q.popleft()
        curr_level = parent_level_q.popleft()
        wbin = (parent_idx) * k + 1
        base_addr = row_lens[curr_level]
        wstart = base_addr + wbin
        print(wstart, end=",")
        if wstart > len(padded_data):
            continue
        wend = min(len(padded_data), wstart + k)
        for i, elem in enumerate(padded_data[wstart:wend]):
            if elem == None:
                continue
            padded_data[wstart + i] = (
                wstart + i
                if padded_data[wstart + i] == 0
                else -1
                if padded_data[wstart + i] != -1
                else padded_data[wstart + i] - 1
            )
            if curr_level + 1 < len(row_lens):
                parent_idx_q.extend([wbin + i])
                parent_level_q.extend([curr_level + 1])
    parent_idx_q.popleft()
    parent_level_q.popleft()
    if len(parent_idx_q):
        parent_idx_q.append(None)
        parent_level_q.append(None)

print(padded_data)
