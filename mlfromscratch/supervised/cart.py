import numpy as np


class Classification_tree (object):
    """
    Single decision tree for classification.

    Parameters
    ----------
    leaf_size: int, default=1
        minimum number of samples required to be at a leaf node.
    """
    def __init__(self, leaf_size=1):
        self.leaf_size = leaf_size
        self.path = path
