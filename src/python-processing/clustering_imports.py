#!/usr/bin/python3
import numpy as np
import os
import sys
import argparse
# from kmedoids import *
# from kmeans import *

# from cluster_evaluation import *
# from geometry_fxns import *
# from TreeNode import TreeNode
# from Datum import Datum
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

# def graham_scan_convex_hull(points):
#     def cross_product(p, q, r):
#         return (q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1])

#     def polar_angle(p, q):
#         return np.arctan2(q[1] - p[1], q[0] - p[0])

#     # Find the point with the lowest y-coordinate (and leftmost if tied)
#     start_point = min(points, key=lambda p: (p[1], p[0]))

#     # Sort the points based on polar angles with respect to the start point
#     sorted_points = sorted(
#         points, key=lambda p: (polar_angle(start_point, p), -p[1], p[0])
#     )

#     # Initialize the convex hull with the first three sorted points
#     convex_hull = [sorted_points[0], sorted_points[1], sorted_points[2]]

#     # Iterate through the sorted points and construct the convex hull
#     for point in sorted_points[3:]:
#         while (
#             len(convex_hull) >= 2
#             and cross_product(convex_hull[-2], convex_hull[-1], point) <= 0
#         ):
#             convex_hull.pop()
#         convex_hull.append(point)

#     return convex_hull
