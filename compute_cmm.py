import plac
from os import path
import numpy as np
from scipy import sparse
from scipy.io import savemat

from cmmlib.inout import load_mesh, save_coff
from cmmlib import cmm


@plac.annotations(
    K=('number of CMHBs', 'positional', None, int),
    mu=('sparsity parameter mu', 'positional', None, float),
    visualize=('visualize the weights?', 'flag', 'v'),
    scaled=('respect triangle scaling?', 'flag', 's'),
    output_dir=('output directory', 'option', 'o'),
    maxiter=('maximum number of iterations', 'option', None, int),
    ply=('output ply file?', 'flag', None),
    off=('output off file?', 'flag', None)
)
def main(input_filename, K, mu, output_dir=None, visualize=False, scaled=False,
         maxiter=None, ply=False, off=False):

    if (off or ply) and not output_dir:
        print "please specify an output directory"
        return 1

    if output_dir and not path.exists(output_dir):
        print "%s does not exist" % output_dir
        return 2

    verts, tris = load_mesh(input_filename, normalize=True)
    print "%d vertices, %d faces" % (len(verts), len(tris))

    Phi_cpr, D = cmm.compressed_manifold_modes(
        verts, tris, K, mu=mu, scaled=scaled,
        maxiter=maxiter, verbose=100, return_D=True)

    if D is None:
        D_diag = np.ones(len(verts))
        D = sparse.eye(len(verts))
    else:
        D_diag = D.data

    if visualize:
        from cmmlib.vis.weights import show_weights
        show_weights(verts, tris, Phi_cpr)

    if output_dir:
        # save in simple text format
        np.savetxt(path.join(output_dir, 'phi.txt'), Phi_cpr, fmt='%f')
        np.savetxt(path.join(output_dir, 'D_diag.txt'), D_diag, fmt='%f')

        # save in matlab format
        savemat(path.join(output_dir, 'phi.mat'),
                dict(verts=verts, tris=tris+1, phi=Phi_cpr, D=D))

        # save HDF5 format if possible
        try:
            import h5py
        except ImportError:
            print "Cannot save as HDF5, please install the h5py module"
        else:
            with h5py.File(path.join(output_dir, 'phi.h5'), 'w') as f:
                f['verts'] = verts
                f['tris'] = tris
                f['phi'] = Phi_cpr
                f['d_diag'] = D_diag

        # save NPY format
        np.save(path.join(output_dir, 'phi.npy'), Phi_cpr)
        np.save(path.join(output_dir, 'D_diag.npy'), Phi_cpr)

        if off or ply:
            # map phi scalars to colors
            from mayavi.core.lut_manager import LUTManager
            from cmmlib.vis.weights import _centered
            lut = LUTManager(lut_mode='RdBu').lut.table.to_array()[:, :3]
            colors = [
                lut[(_centered(Phi_cpr[:, k]) * (lut.shape[0]-1)).astype(int)]
                for k in xrange(K)]
            # save in a single scene as a collage
            w = int(np.ceil(np.sqrt(K))) if K > 6 else K
            spacing = 1.2 * verts.ptp(axis=0)
            all_verts = [verts + spacing * (1.5, 0, 0)]
            all_tris = [tris]
            all_color = [np.zeros(verts.shape, np.int) + 127]
            for k in xrange(K):
                all_verts.append(verts + spacing * (-(k % w), 0, int(k / w)))
                all_tris.append(tris + len(verts) * (k+1))
                all_color.append(colors[k])

            if off:
                save_coff(path.join(output_dir, 'input.off'),
                          verts.astype(np.float32), tris)
                for k in xrange(K):
                    save_coff(path.join(output_dir, 'cmh_%03d.off' % k),
                              verts.astype(np.float32), tris, colors[k])
                save_coff(path.join(output_dir, 'all.off'),
                          np.vstack(all_verts), np.vstack(all_tris),
                          np.vstack(all_color))

            if ply:
                from tvtk.api import tvtk
                pd = tvtk.PolyData(
                    points=np.vstack(all_verts).astype(np.float32),
                    polys=np.vstack(all_tris).astype(np.uint32))
                pd.point_data.scalars = np.vstack(all_color).astype(np.uint8)
                pd.point_data.scalars.name = 'colors'
                ply = tvtk.PLYWriter(
                    file_name=path.join(output_dir, 'all.ply'),
                    input=pd, color=(1, 1, 1))
                ply.array_name = 'colors'
                ply.write()


if __name__ == '__main__':
    plac.call(main)
