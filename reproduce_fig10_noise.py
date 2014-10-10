import numpy as np

from cmmlib.util import compute_average_edge_length
from cmmlib.inout import load_mesh
from cmmlib import cmm
from cmmlib.vis.weights import WeightsVisualization


def main():
    experiments = [
        ('meshes/fertility/FE_20k.off', 6, [0., 0.25, 0.5, 1.0], 12.5),
        ('meshes/bimba/bimba_cvd_10k.off', 5, [0, 0.25, 0.5, 1.0], 30),
    ]

    wvs = []

    for filename, K, noise_levels, mu in experiments:
        verts, tris = load_mesh(filename, check=False)
        verts = verts / verts.std()
        avg_edge_len = compute_average_edge_length(verts, tris)

        # compute cmm for each noise level
        vis_meshes = []
        for noise_level in noise_levels:
            if noise_level <= 0:
                verts_noisy = verts
            else:
                noise_scale = noise_level * avg_edge_len
                noise = np.random.normal(scale=noise_scale, size=verts.shape)
                verts_noisy = verts + noise

            Phi_cmh = cmm.compressed_manifold_modes(
                verts_noisy, tris, K,
                mu=mu, scaled=True, init='varimax',
            )

            if noise_level <= 0:
                label = 'no noise'
            else:
                label = '%.0f%% gaussian noise' % (noise_level * 100)

            vis_meshes.append((verts_noisy, tris, Phi_cmh, label))

        # initialize the visualization, show later when all experiments done
        wvs.append(WeightsVisualization(
            vis_meshes, show_labels=True,
            actor_options=dict(interpolation='flat')))

    # show visualization windows
    for wv in wvs[:-1]:
        wv.edit_traits()
    wvs[-1].configure_traits()

if __name__ == '__main__':
    main()
