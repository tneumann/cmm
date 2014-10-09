from itertools import product
import numpy as np

from cmmlib import cmm
from cmmlib.inout import load_mesh
from cmmlib.vis.weights import show_weights


K = 1
num_inits = 6
at_iters = [0, 2, 10, 50, 200, 1000]
verts, tris = load_mesh('meshes/bunny_fixed.obj')
_, _, color1 = load_mesh('meshes/bunny_fixed_ear.off', no_colors=False)
_, _, color2 = load_mesh('meshes/bunny_fixed_sgp.off', no_colors=False)

all_Phis = []
for init in xrange(num_inits):
    if init == 0:
        Phi_init = -1 * (color1[:, 0] > 200).astype(np.float).reshape(len(verts), K)
    elif init == 1:
        Phi_init = 1 * (color2[:, 0] > 200).astype(np.float).reshape(len(verts), K)
    else:
        Phi_init = np.random.uniform(-1, 1, (len(verts), K))
    Phis = [Phi_init]

    def callback(H, mu, Phi, E, S, **kwargs):
        global i
        i += 1
        if i in at_iters:
            Phis.append(Phi)

    i = 0
    cmm.compressed_manifold_modes(
        verts, tris, K, mu=20, init=Phi_init, scaled=True,
        callback=callback, tol_abs=0, tol_rel=0, maxiter=max(at_iters)+1,
        check_interval=1)

    all_Phis.append(Phis)

Phi_at_iters = np.array([np.array(Phis) for Phis in all_Phis])

show_weights(
    verts, tris,
    Phi_at_iters.reshape((num_inits*len(at_iters), len(verts), K)),
    ["Iteration %d (%d)" % (it, exp) if it > 0 else "Initialization (%d)" % exp
     for exp, it in product(range(num_inits), at_iters)],
    show_labels=True, label_offset_axis=1, offset_spacing2=1.3, 
    label_offset=0.8, num_columns=len(at_iters),
)
