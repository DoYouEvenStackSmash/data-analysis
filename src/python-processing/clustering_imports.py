#!/usr/bin/python3
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import dendrogram, linkage

colors = list(set(mcolors.CSS4_COLORS))
from render_support import *


def dataloader(filename):
    """
    Loads data from a file
    """
    # Check if the file exists
    if not os.path.exists(filename):
        print("File does not exist.")
        sys.exit()
    else:
        # Load the array using np.load()
        loaded_array = np.load(filename, allow_pickle=True)
        # Display the loaded array
        print("Loaded Array:")
        print(loaded_array.shape)
        return loaded_array
