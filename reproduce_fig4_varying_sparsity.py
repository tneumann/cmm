import numpy as np

from cmmlib import cmm
from cmmlib.align import optimal_permutation
from cmmlib.inout import load_mesh
from cmmlib.vis.weights import show_weights


verts, tris = load_mesh('meshes/hand_4054.obj')
K = 6
mus = [1/100., 1., 10.]

all_Phi = [cmm.manifold_harmonics(verts, tris, K)] \
        + [cmm.compressed_manifold_modes(verts, tris, K, mu, scaled=False)
           for mu in mus]

# permute modes so we can visualize them side-by-side
all_Phi_aligned = \
    [np.dot(optimal_permutation(Phi.T, all_Phi[-1].T, allow_reflection=True), Phi.T).T
     for Phi in all_Phi[:-1]] + \
    [all_Phi[-1]]

show_weights(verts, tris, all_Phi_aligned, names=['dense'] + map(str, mus))
