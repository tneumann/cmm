from cmmlib import cmm
from cmmlib.inout import load_mesh
from cmmlib.vis.weights import show_many_weights


mu = 2.0
K = 6

vms = []
for i, fn in enumerate(['hand_868', 'hand_868_holes2']):
    verts, tris = load_mesh('meshes/%s.obj' % fn)

    Phi_cpr = cmm.compressed_manifold_modes(verts, tris, K, mu=mu)
    Phi_dense = cmm.manifold_harmonics(verts, tris, K)

    vms += [(verts, tris, Phi_cpr, 'CMM #%d' % i),
            (verts, tris, Phi_dense, 'MH #%d' % i)]

show_many_weights(vms, show_labels=True)
