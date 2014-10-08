import numpy as np
from scipy.spatial.distance import cdist


def optimal_linear_map(frompts, topts):
    C = np.linalg.solve(np.dot(frompts, frompts.T), np.dot(frompts, topts.T)).T
    return C, np.dot(C, frompts)


def optimal_rotation(frompts, topts, allow_reflection=False):
    U, s, Vt = np.linalg.svd(np.dot(topts, frompts.T))
    if not allow_reflection and np.linalg.det(np.dot(U, Vt)) < 0:
        Vt[-1] *= -1
    R = np.dot(U, Vt)
    return R


def optimal_permutation(frompts, topts, allow_reflection=False, 
                        distance_metric='euclidean'):
    # compute distances of points
    D = cdist(frompts, topts, distance_metric)
    if allow_reflection:
        Dreflection = cdist(frompts, -1 * topts)
        reflection_sign = np.where(Dreflection < D, -1, 1)
        D = np.minimum(Dreflection, D)
    else:
        reflection_sign = np.ones(frompts.shape[0])
    Dmax = D.max()

    # construct bipartite graph
    import networkx as nx
    graph = nx.Graph()
    for i in xrange(frompts.shape[0]):
        graph.add_node("A%d" % i)
        graph.add_node("B%d" % i)
    for i in xrange(frompts.shape[0]):
        for j in xrange(frompts.shape[0]):
            w = Dmax - D[i, j]
            graph.add_edge("A%d" % i, "B%d" % j, {'weight': w})
    # compute max matching of graph
    mates = nx.max_weight_matching(graph, True)
    # fill permutation matrix according to matching result
    R = np.zeros((frompts.shape[0], topts.shape[0]))
    for k, v in mates.iteritems():
        if k.startswith('A'):
            i, j = int(k[1:]), int(v[1:])
            R[i, j] = reflection_sign[i, j] if allow_reflection else 1

    R = R.T  # mapping frompts-topts
    return R


def optimal_permutation_greedy(frompts, topts, allow_reflection=True, 
                               distance_metric='euclidean'):
    D = cdist(frompts, topts, distance_metric)
    is_reflection = np.zeros(D.shape, np.bool)
    if allow_reflection:
        D_reflect = cdist(frompts, -1 * topts, distance_metric)
        is_reflection = D_reflect < D
        D = np.minimum(D, D_reflect)
    D = D.max() - D
    K = topts.shape[0]
    C = np.zeros((frompts.shape[0], topts.shape[0]))
    for i in xrange(frompts.shape[0]):
        best = D[i].argmax()
        if np.isinf(D[i, best]):
            continue
        v = -1.0 if is_reflection[i, best] else 1.0
        D[:, best] = -np.inf
        C_row = np.zeros(K)
        C_row[best] = v
        C[i] = C_row
    return np.array(C).T


def closest_rotation_matrix(C, allow_reflection=False):
    U, s, Vt = np.linalg.svd(C)
    if not allow_reflection and np.linalg.det(np.dot(U, Vt)) < 0:
        Vt[-1] *= -1
    return np.dot(U, Vt)


def optimal_reflection(a, b):
    R = np.eye(a.shape[0])
    for i in xrange(a.shape[0]):
        if np.linalg.norm(a[i] - b[i]) > np.linalg.norm(-a[i] - b[i]):
            R[i, i] = -1
    return R


def test_optimal_permutation():
    d = 10
    a = np.random.random((d, 500))
    for i in xrange(100):
        shuffle = np.random.permutation(d)
        R_true = np.zeros((d, d))
        R_true[np.arange(d), shuffle] = np.random.choice((-1, 1), len(a))
        b = np.dot(R_true, a)
        R = optimal_permutation(a, b, allow_reflection=True)
        np.testing.assert_array_equal(R, R_true)
        a_rot = np.dot(R, a)
        np.testing.assert_array_equal(a_rot, b)


def test_optimal_linear_map():
    for i in xrange(100):
        d = np.random.randint(2, 20)
        a = np.random.random((d, 500))
        C_true = np.random.normal(size=(d, d))
        b = np.dot(C_true, a)
        C, a_transformed = optimal_linear_map(a, b)
        np.testing.assert_allclose(a_transformed, b)
        np.testing.assert_allclose(np.dot(C, a), b)
        np.testing.assert_allclose(C, C_true)
        C_inv = np.linalg.inv(optimal_linear_map(b, a)[0])
        np.testing.assert_allclose(C, C_inv)


def test_optimal_rotation():
    for i in xrange(100):
        d = np.random.randint(2, 20)
        a = np.random.random((d, 500))
        R_true = np.linalg.qr(np.random.normal(size=(d, d)))[0]
        b = np.dot(R_true, a)
        R = optimal_rotation(a, b, allow_reflection=True)
        np.testing.assert_allclose(np.dot(R, a), b)
        np.testing.assert_allclose(R, R_true)

    for i in xrange(100):
        d = np.random.randint(2, 20)
        a = np.random.random((d, 500))
        R_true = np.linalg.qr(np.random.normal(size=(d, d)))[0]
        if np.linalg.det(R_true) < 0:
            R_true[-1] *= -1
        b = np.dot(R_true, a)
        R = optimal_rotation(a, b, allow_reflection=False)
        np.testing.assert_allclose(np.dot(R, a), b)
        np.testing.assert_allclose(R, R_true)


if __name__ == '__main__':
    test_optimal_permutation()
    test_optimal_linear_map()
    test_optimal_rotation()
