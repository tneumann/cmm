import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from cmmlib.util import filter_reindex
from cmmlib.inout import load_mesh
from cmmlib import cmm
from cmmlib.vis.weights import WeightsVisualization


def main():
    experiments = [
        ('meshes/fertility/FE_20k.off', 6, [0., 500], 12.5),
        ('meshes/bimba/bimba_cvd_10k.off', 5, [0, 100, 200], 30),
    ]

    wvs = []

    for filename, K, all_n_holes, mu in experiments:
        verts, tris = load_mesh(filename, check=False)
        verts = verts / verts.std()

        # compute cmm for each noise level
        vis_meshes = []
        for n_holes in all_n_holes:
            if n_holes <= 0:
                tris_holes = tris
                verts_holes = verts
            else:
                while True:
                    # place holes
                    hole_centers = np.random.randint(0, len(verts), n_holes)
                    keep_vertex = np.ones(len(verts), np.bool)
                    keep_vertex[hole_centers] = False
                    keep_tri = keep_vertex[tris].all(axis=1)
                    tris_holes = filter_reindex(keep_vertex, tris[keep_tri])
                    verts_holes = verts[keep_vertex]
                    # check if mesh is still a single connected graph
                    ij = np.r_[np.c_[tris_holes[:, 0], tris_holes[:, 1]],
                               np.c_[tris_holes[:, 0], tris_holes[:, 2]],
                               np.c_[tris_holes[:, 1], tris_holes[:, 2]]]
                    n_keep = verts[keep_vertex].shape[0]
                    G = csr_matrix(
                        (np.ones(len(ij)), ij.T),
                        shape=(n_keep, n_keep))
                    n_components, labels = connected_components(G, directed=0)
                    if n_components == 1:
                        break
                    else:
                        # mesh was torn apart by hole creation process 
                        # trying another set of random holes
                        pass

            Phi_cmh = cmm.compressed_manifold_modes(
                verts_holes, tris_holes, K, 
                mu=mu, scaled=True, init='varimax',
            )

            if n_holes <= 0:
                label = 'no holes'
            else:
                label = '%d holes' % (n_holes)

            vis_meshes.append((verts_holes, tris_holes, Phi_cmh, label))

        # initialize the visualization, show later when all experiments done
        wvs.append(WeightsVisualization(vis_meshes, show_labels=True))

    # show visualization windows
    for wv in wvs[:-1]:
        wv.edit_traits()
    wvs[-1].configure_traits()

if __name__ == '__main__':
    main()
