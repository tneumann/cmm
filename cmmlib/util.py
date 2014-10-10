import numpy as np


def veclen(vectors):
    """ return L2 norm (vector length) along the last axis, for example to compute the length of an array of vectors """
    return np.sqrt(np.sum(vectors**2, axis=-1))


def filter_reindex(condition, target):
    """
    Filtering of index arrays.

    To explain this, let me give an example. Let's say you have the following data:
    >>> data = np.array(['a', 'b', 'c', 'd'])
    
    You also have another array that consists of indices into this array
    >>> indices = np.array([0, 3, 3, 0])
    >>> data[indices].tolist()
    ['a', 'd', 'd', 'a']

    Now, say you are filtering some elements in your data array
    >>> condition = (data == 'a') | (data == 'd')
    >>> filtered_data = data[condition]
    >>> filtered_data.tolist()
    ['a', 'd']

    The problem is that your index array doesn't correctly reference the new filtered data array
    >>> filtered_data[indices]
    Traceback (most recent call last):
        ...
    IndexError: index 3 is out of bounds for size 2

    Based on an old index array (target), this method returns a new index array 
    that re-indices into the data array as if condition was applied to this array, so
    >>> filtered_indices = filter_reindex(condition, indices)
    >>> filtered_indices.tolist()
    [0, 1, 1, 0]
    >>> filtered_data[filtered_indices].tolist()
    ['a', 'd', 'd', 'a']

    >>> indices = np.array([1, 4, 1, 4])
    >>> condition = np.array([False, True, False, False, True])
    >>> filter_reindex(condition, indices).tolist()
    [0, 1, 0, 1]
    """
    if condition.dtype != np.bool:
        raise ValueError, "condition must be a binary array"
    reindex = np.cumsum(condition) - 1
    return reindex[target]


def compute_average_edge_length(verts, tris):
    edges = tris[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    edges = np.sort(edges, axis=1)
    ij_dtype = [('i', edges.dtype), ('j', edges.dtype)]
    edges_uniq = np.unique(edges.view(dtype=ij_dtype))
    edges_uniq = edges_uniq.view(dtype=edges.dtype).reshape(-1, 2)
    return veclen(verts[edges_uniq[:, 0]] - verts[edges_uniq[:, 1]]).mean()
