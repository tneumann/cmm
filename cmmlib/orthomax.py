import numpy as np
from scipy.linalg import svd

__all__ = ['orthomax']

def orthomax(Phi, gamma=1., maxiter=20, tol=1e-6):
    """
    Given n*k factor matrix Phi (k number of factors, n number of dimensions),
    find a rotated version of the factor loadings that fulfills optimization criteria, 
    depending on gamma \in [0..1]

    If gamma=0 this is the "quartimax", for gamma=1 it is often called "varimax".

    It is not easy to find a good reference on this,
    one is 
        Herve Abdi, "Factor Rotations in Factor Analyses."
        http://www.utd.edu/~herve/Abdi-rotations-pretty.pdf
    another which also has an algorithm listing and a nice application, but only for gamma = 0
        Justin Solomon et al., "Discovery of Intrinsic Primitives on Triangle Meshes", Eurographics 2011
        http://graphics.stanford.edu/projects/lgl/papers/sbbg-diptm-11/sbbg-diptm-11.pdf
    """
    if gamma < 0 or gamma > 1:
        raise ValueError, "gamma must be between 0 and 1"
    p, k = Phi.shape
    R = np.eye(k)
    Lambda = Phi
    d_old = None
    for i in xrange(maxiter):
        if gamma > 0:
            Z = Lambda**3 - (gamma / p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T, Lambda))))
        else:
            Z = Lambda**3
        U, s, Vt = svd(np.dot(Phi.T, Z), full_matrices=False)
        R = np.dot(U, Vt)
        Lambda = np.dot(Phi, R)
        d = np.sum(s)
        if d_old != None and d < d_old * (1 + tol):
            print "orthomax converged"
            break
        d_old = d
    return np.dot(Phi, R)


if __name__ == '__main__':
    import pylab as pl

    # generate randomly activated 2d gaussian blobs
    np.random.seed(2)
    n = 16
    n_components = 3
    n_obs = 200
    y, x = np.mgrid[:n, :n]
    xy = np.dstack((x, y)).astype(np.float)
    components = []
    for i in xrange(n_components):
        mean = np.random.uniform(0, n, 2)
        sigma = np.random.uniform(0, n/3, 2)
        p = np.exp( - ((xy - mean)**2 / (2*sigma)).sum(axis=-1))
        components.append(p.ravel())
    components = np.array(components)
    code = np.random.random((n_obs, n_components)) \
            + np.random.random(n_obs)[:,np.newaxis]
    obs = np.dot(code, components)
    obs += np.random.normal(scale=0.2, size=obs.shape)

    # pca
    U, s, Vt = svd(obs)
    pca_components = Vt[:n_components, :]
    # orthomaximalize the pca factors
    vmax = orthomax(pca_components.T, gamma=1., maxiter=100)

    # visualize
    pl.clf()
    for i, c in enumerate(pca_components):
        pl.subplot(3, n_components, i+1)
        pl.title("pca#%d" % i)
        pl.imshow(c.reshape(n,n))
    for i, c in enumerate(vmax.T):
        pl.subplot(3, n_components, i+4)
        pl.title("orthomax#%d" % i)
        pl.imshow(c.reshape(n,n))
    for i, c in enumerate(components[[2,0,1]]):
        pl.subplot(3, n_components, i+7)
        pl.title("original#%d" % i)
        pl.imshow(c.reshape(n,n))
    pl.show()

