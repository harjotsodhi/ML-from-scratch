import numpy as np
from mlfromscratch.helpers import helpers as hp



class Classification_tree (object):
    """
    Single decision tree for classification implemented recursively.

    Parameters
    ----------
    leaf_size: int, default=1
        minimum number of samples required to be at a leaf node.
    """
    def __init__(self, leaf_size=1):
        self.leaf_size = leaf_size
        # path structure, each node either:
        # root: {col_index:int,val:float,leftTree:dict,rightTree:dict}
        # or
        # leaf: {class:int/float/str}
        self.path = {}


    def fit(self, X, y):
        """
        Fit decision tree.

        Parameters
        ----------
        X: np.array
            feature matrix (m x n)

        y: np.array
            response vector (m x 1)
        """
        path = self._build_tree(X, y)
        self.path = path


    def predict(self, X):
        '''
        Return the predicted value.

        Parameters
        ----------
        X: np.array
            feature matrix (m x n)
        '''
        y_pred = np.zeros(X.shape[0])

        for r in range(X.shape[0]):
            y_pred[r] = self._dfs(X[r,:])

        return y_pred


    #### Private methods ###

    def _build_tree(self, X, y):
        """
        Recursively build tree

        Parameters
        ----------
        X: np.array
            feature matrix (m x n)

        y: np.array
            response vector (m x 1)
        """
        # base condition 1: number of samples <= self.leaf_size, return leaf
        if y.shape[0] <= self.leaf_size:
            y_pred = hp.mode(y)
            return {'class':y_pred}

        # base condition 2: all samples belong to same class, return leaf
        if hp.single_class(y):
            # for higher values of leaf_size, there may not be a single class
            # so the mode is used
            y_pred = y[0]
            return {'class':y_pred}

        # make the best split
        r_ind, c_ind = self._split(X, y)

        # build left tree
        m_left = np.flatnonzero(X[:,c_ind]>X[r_ind,c_ind])
        leftTree = self._build_tree(X[m_left], y[m_left])

        # build right tree
        m_right = np.flatnonzero(X[:,c_ind]<=X[r_ind,c_ind])
        rightTree = self._build_tree(X[m_right], y[m_right])

        return {'col_index':c_ind,'row_val':X[r_ind,c_ind],'left':leftTree,'right':rightTree}


    def _split(self, X, y):
        """
        Calculate the best split feature and value w.r.t information gain.

        Best feature i and best value j = Argmax(Information Gain)
        Information Gain = Entropy(y) - Entropy(y|X_ij)

        Find the best feature i and value j in X such that
            Entropy(y) - Entropy(y|X_ij) is maximized

        Parameters
        ----------
        X: np.array
            feature matrix (m x n)

        y: np.array
            response vector (m x 1)
        """
        e_y = self._entropy(y)
        e_x = np.zeros_like(X, dtype=float)

        # iterate through each element in the matrix
        for c in range(X.shape[1]):
            for r in range(X.shape[0]):
                # split column c based on row r
                m_left = np.flatnonzero(X[:,c]>X[r,c])
                m_right = np.flatnonzero(X[:,c]<=X[r,c])
                # weighted average of both nodes
                total = y[m_left].shape[0] + y[m_right].shape[0]
                e_left = self._entropy(y[m_left]) * y[m_left].shape[0]/total
                e_right = self._entropy(y[m_right]) * y[m_right].shape[0]/total
                e_x[r,c] = e_left+e_right

        # matrix of information gained
        ig_mat = e_y - e_x
        # max information gain row and col index
        r_ind, c_ind = np.unravel_index(np.argmax(ig_mat), ig_mat.shape)
        return r_ind, c_ind


    def _dfs(self, X, node=None):
        """
        Recursively find class prediction for X
            via DFS of the fitted path.

        Parameters
        ----------
        X: np.array
            feature matrix (m x n)
        """
        # root node
        if not node:
            node = self.path

        # base condition
        if 'class' in node:
            return node['class']

        # traverse left or right depending on condition
        if X[node['col_index']]>node['row_val']:
            y_pred = self._dfs(X, node=node['left'])
        else:
            y_pred = self._dfs(X, node=node['right'])

        return y_pred


    def _entropy(self, arr, normalize=False):
        """
        Calculate the entropy of an array.

        Entropy = expected value of surprise
        Surprise = log(1/p(x))

        Parameters
        ----------
        arr: np.array
            vector (m x 1)

        normalize: bool, default=False
            whether to normalize entropy
            only relevant if number of classes > 2
        """
        vals,counts = np.unique(arr, return_counts=True)
        # probability of each class
        p = counts/counts.sum()
        # surprise of each class
        s = np.log2(p)
        # expected value of surprise
        e = -(p*s).sum()
        # normalize option
        if normalize:
            return e/np.log2(p.shape[0])

        return e
