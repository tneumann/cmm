from cmmlib import cmm
from cmmlib.inout import load_mesh
from cmmlib.vis.weights import show_weights


K1 = 8 + 6 + 12
K2 = 4
verts, tris = load_mesh('meshes/bumpy_cube6.obj')

Phi_cpr1 = cmm.compressed_manifold_modes(verts, tris, K1, mu=20., scaled=False)
Phi_cpr2 = cmm.compressed_manifold_modes(verts, tris, K2, mu=20., scaled=False)
Phi_vari1 = cmm.varimax_modes(verts, tris, K1)
Phi_vari2 = cmm.varimax_modes(verts, tris, K2)

# establish some consistent ordering of varimax modes just for visualization
i = (Phi_cpr2[:, 0]**2).argmax()  # select one vertex
# permute according to activation strength of that vertex
Phi_vari2 = Phi_vari2[:, (Phi_vari2[i]**2).argsort()[::-1]]
Phi_vari1 = Phi_vari1[:, (Phi_vari1[i]**2).argsort()[::-1]]
Phi_cpr2 = Phi_cpr2[:, (Phi_cpr2[i]**2).argsort()[::-1]]
Phi_cpr1 = Phi_cpr1[:, (Phi_cpr1[i]**2).argsort()[::-1]]

show_weights(
    verts, tris,
    (Phi_cpr1, Phi_cpr2, Phi_vari1, Phi_vari2),
    ('CMM, K=%d' % K1, 'CMM, K=%d' % K2, 'Variax, K=%d' % K1, 'Varimax, K=%d' % K2),
    contours=7, show_labels=True)
