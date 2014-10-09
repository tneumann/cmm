import numpy as np
from os import path
from mayavi.core.lut_manager import LUTManager

from cmmlib import cmm
from cmmlib.inout import load_mesh
from cmmlib.vis.weights import show_many_weights



experiments = [
    ('meshes/armadillo_61372.obj', [3, 7], 20.),
    ('meshes/elk.off', [3, 7], 20.),
]
results = []

for filename, Ks, mu in experiments:
    verts, tris = load_mesh(filename, normalize=True)

    # for different K, compute the CMMs
    Phis = []
    for K in Ks:
        Phi_cmh = cmm.compressed_manifold_modes(
            verts, tris, K, mu, scaled=True, init='varimax')
        Phis.append(Phi_cmh)

    # color the mesh according to the CMMs
    lut = LUTManager(lut_mode='hsv').lut.table.to_array()
    colors = []
    for K, Phi in zip(Ks, Phis):
        # pass Phi through the lookup table depending on it's strength
        ti = np.linspace(0, 1, K, endpoint=False) * (lut.shape[0]-1)
        lut_rgb = lut[ti.astype(np.int), :3].astype(np.float)
        Phi = np.abs(Phi / (Phi.max(0) - Phi.min(0))[np.newaxis])
        a = Phi.sum(axis=1)[:, np.newaxis]
        Phi_color = (lut_rgb[np.newaxis] * Phi[:, :, np.newaxis]).sum(axis=1)
        # mix with grey
        color = np.clip(Phi_color * a + 200 * (1-a), 0, 255)
        colors.append(color)

    results.append((verts, tris, colors))

# visualize!
show_many_weights([
    (verts, tris, color.astype(np.uint8)[...,np.newaxis], "#%s K=%d" % (path.basename(fn), K))
    for (fn, _, _), (verts, tris, colors) in zip(experiments, results)
    for K, color in zip(Ks, colors)],
    center=False, show_labels=True,
    )
