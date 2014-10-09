from cmmlib import cmm
from cmmlib.inout import load_mesh
from cmmlib.vis.weights import show_weights


K = 8 + 6 + 12
mu = 20.0
filename = 'meshes/bumpy_cube6.obj'

verts, tris = load_mesh(filename)

Phi_cpr = cmm.compressed_manifold_modes(verts, tris, K, mu=mu)
Phi_dense = cmm.manifold_harmonics(verts, tris, K)
Phi_vari = cmm.varimax_modes(verts, tris, K)

show_weights(verts, tris, (Phi_cpr, Phi_dense, Phi_vari),
             ('CMM', 'MH', 'Varimax'), show_labels=True)
