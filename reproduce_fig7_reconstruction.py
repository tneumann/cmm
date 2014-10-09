import numpy as np
from cmmlib import cmm
from cmmlib.inout import load_mesh
from cmmlib.vis.weights import show_many_weights


K = 6
verts, tris = load_mesh('meshes/hand_4054.obj')

# compute bases
Phi_cmh = cmm.compressed_manifold_modes(verts, tris, K, mu=2, scaled=False)
Phi_mh = cmm.manifold_harmonics(verts, tris, K, scaled=False)

# reconstruct from bases
verts_rec_cmh = np.dot(Phi_cmh, np.dot(Phi_cmh.T, verts))
verts_rec_mh = np.dot(Phi_mh, np.dot(Phi_mh.T, verts))

show_many_weights(
    ((verts, tris, None, 'Input'),
     (verts_rec_mh, tris, None, 'MH reconstr.'),
     (verts_rec_cmh, tris, None, 'CMM reconstr.')),
    show_labels=True,
    actor_options=dict(edge_visibility=True, line_width=1.0))
