#!/usr/bin/python3
import numpy as np

import torch

from filter_gen import *

# load coordinates
ps = np.load("/home/aroot/stuff/fft/posStruc.npy")
ps.shape
tps = torch.from_numpy(ps)

grid = gen_grid(128, 0.3).reshape(-1, 1)
# preprocessing
tps = generate_rotations(tps, 10)

# simulate images
imcpu = simulate_images(tps, grid, 1.0)

# generate ctfs
ctf_batch = generate_ctfs(10)

tps.shape
n_pixel = 128
pixel_size = 0.3
params = {
    "amp": 0.1,
    "b_factor": 1.0,
    "defocus": {"min": 0, "max": 1, "step": 0.1},
    "n_pixel": 128,
    "pixel_size": 0.3,
}

# ctf_batch = generate_ctfs(10)
params = {
    "amp": 0.1,
    "b_factor": 1.0,
    "defocus": {"min": 0, "max": 1, "step": 0.1},
    "n_pixel": 128,
    "pixel_size": 0.3,
}
# np.save("ctfs.npy",)
print(imcpu)
