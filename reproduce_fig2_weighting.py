import numpy as np

from cmmlib import cmm
from cmmlib.inout import load_mesh
from cmmlib.vis.weights import WeightsVisualization


# load the bumpy meshes in 2 resolutions
verts1, tris1 = load_mesh('meshes/bump512.obj')
verts2, tris2 = load_mesh('meshes/bump32768.obj')

# we need to use a trick here to obtain the mode on the bump itself
# initialize the manifold mode such that it is zero at every vertex, 
# but one at the peak of the bump
Phi_init1 = np.zeros((len(verts1), 1))
Phi_init1[verts1[:, 1].argmax()] = 1.0
Phi_init2 = np.zeros((len(verts2), 1))
Phi_init2[verts2[:, 1].argmax()] = 1.0

# set parameters
# NOTE: these might not give exactly the same results as in the paper
#       since the code was significantly changed after producing Fig. 2
#       the code here still shows the same behavior of the scaling
K = Phi_init1.shape[1]
mu_scaled = 50.
mu_unscaled = 20.0
params = dict(maxiter=5000)

# compute modes
Phi1_scaled = cmm.compressed_manifold_modes(
    verts1, tris1, K, mu_scaled,
    init=Phi_init1, scaled=True, **params)

Phi2_scaled = cmm.compressed_manifold_modes(
    verts2, tris2, K, mu_scaled,
    init=Phi_init2, scaled=True, **params)

Phi1_unscaled = cmm.compressed_manifold_modes(
    verts1, tris1, K, mu_unscaled,
    init=Phi_init1, scaled=False, **params)

Phi2_unscaled = cmm.compressed_manifold_modes(
    verts2, tris2, K, mu_unscaled,
    init=Phi_init2, scaled=False, **params)

# visualize
wv = WeightsVisualization([
    (verts1, tris1, None, 'wireframe1'),
    (verts2, tris2, None, 'wireframe2'),
    (verts1, tris1, Phi1_unscaled, 'unscaled1'),
    (verts2, tris2, Phi2_unscaled, 'unscaled2')
    (verts1, tris1, Phi1_scaled, 'scaled1'),
    (verts2, tris2, Phi2_scaled, 'scaled2'),
], contours=5)

wv._trimeshes[0][0].actor.property.set(edge_visibility=True, line_width=1)
wv._trimeshes[1][0].actor.property.set(edge_visibility=True, line_width=1)

wv.configure_traits()
