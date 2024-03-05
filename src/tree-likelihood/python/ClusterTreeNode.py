#!/usr/bin/python3


class ClusterTreeNode:
    """An abstraction for use in the hierarchical clustering"""

    def __init__(
        self, val, val_idx=-1, children=None, data=None, params=None, cluster_radius=0
    ):
        self.val = val
        self.val_idx = val_idx
        self.children = children
        self.data_refs = data
        self.param_refs = params
        self.cluster_radius = cluster_radius
