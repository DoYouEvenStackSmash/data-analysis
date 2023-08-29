#!/usr/bin/python3

class ClusterTreeNode:
    def __init__(self, val, val_idx=-1, children=None, data=None):
        self.val = val
        self.val_idx = val_idx
        self.children = children
        self.data_refs = data