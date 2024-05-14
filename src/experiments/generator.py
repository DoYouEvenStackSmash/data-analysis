import numpy as np


def generate_kt(data, k=2):
    d = 1
    while k ** (d - 1) < len(data):
        d += 1
    kterms = np.arange(d)
    terms = [0] * d
    a = 1
    r = k
    for i in range(d):
        terms[i] = np.sum(a * r ** kterms[0 : i + 1])
    return terms


from collections import deque

data = np.random.rand(1000)
k = 2

kt = generate_kt(data, k)
padded_data = [None] * (len(data) + (kt[-1] - len(data)))

print(kt)
level_q = deque([0, None])
parent_q = deque([0, None])

while len(level_q):
    while level_q[0] != None:
        parent_idx = parent_q.popleft()
        curr_level = level_q.popleft()

        base_addr = kt[curr_level] - 1
        winb = parent_idx * k
        wstart = base_addr + winb
        wend = min(wstart + k, len(data))
        for i, elem in enumerate(data[wstart:wend]):
            if elem == None:
                continue
            padded_data[wstart + i] = (
                wstart + i if padded_data[wstart + i] == None else -1
            )
            parent_q.append(winb + i)
            level_q.append(curr_level + 1)
    level_q.popleft()
    parent_q.popleft()

    if len(level_q):
        level_q.append(None)
        parent_q.append(None)

print(padded_data)
