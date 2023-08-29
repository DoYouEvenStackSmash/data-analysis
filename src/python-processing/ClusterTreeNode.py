#!/usr/bin/python3

class ClusterTreeNode:
    def __init__(self, val, children=None, data=None):
        self.val = val
        self.val_idx = -1
        self.children = children
        self.data_refs = data