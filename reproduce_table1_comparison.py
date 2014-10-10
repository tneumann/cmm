import numpy as np
from time import time
from scipy import sparse

from cmmlib.inout import load_mesh
from cmmlib import cmm


def sizeof_fmt(num):
    for x in ['bytes', 'KB', 'MB', 'GB']:
        if num < 1024.0:
            return "%3.1f%s" % (num, x)
        num /= 1024.0
    return "%3.1f%s" % (num, 'TB')


def main():
    meshes = [
        ('hand', 5, 'meshes/hand_868.obj', [10, 30], False),
        ('fertility', 5, 'meshes/fertility/FE_10k.off', [10, 40], False),
        ('bunny', 1.25, 'meshes/bunny_fixed.obj', [10, 40], False),
    ]

    result_lines = []

    for mesh_name, mu, filename, Ks, scaled in meshes:
        verts, tris = load_mesh(filename, check=False, normalize=True)
        print mesh_name, len(verts), len(tris)
        for K in Ks:
            t_cmh = time()
            Phi_cmh, D_cmh = cmm.compressed_manifold_modes(
                verts, tris, K, init='varimax',
                mu=mu, scaled=scaled, return_D=True)
            t_cmh = time() - t_cmh

            t_mh = time()
            Phi_mh, D_mh = cmm.manifold_harmonics(
                verts, tris, K, return_D=True, scaled=scaled)
            t_mh = time() - t_mh

            Phi_mh = Phi_mh.astype(np.float64)
            Phi_cmh = Phi_cmh.astype(np.float64)
            Phi_cmh[np.abs(Phi_cmh) < 1.e-7] = 0
            Phi_cmh_sp = sparse.csr_matrix(Phi_cmh, dtype=np.float64)

            if scaled:
                verts_rec_cmh = np.dot(Phi_cmh, np.dot((D_cmh * Phi_cmh).T, verts))
                verts_rec_mh = np.dot(Phi_mh, np.dot((D_mh * Phi_mh).T, verts))
            else:
                verts_rec_cmh = np.dot(Phi_cmh, np.dot(Phi_cmh.T, verts))
                verts_rec_mh = np.dot(Phi_mh, np.dot(Phi_mh.T, verts))

            line_format = "{mesh_name} & {num_verts:>8d} & {basis} & {K:d} & {mu:>4} & {error:>6.2f} & {size:>8} & {time:>5.2f}s"
            result_lines.append(line_format.format(
                mesh_name=mesh_name, basis='MHB', K=K, mu='-',
                num_verts=len(verts),
                error=np.linalg.norm(verts_rec_mh - verts),
                size=sizeof_fmt(Phi_mh.nbytes),
                time=t_mh,
            ))

            result_lines.append(line_format.format(
                mesh_name=mesh_name, basis='CMB', K=K, mu='%.2f' % mu,
                num_verts=len(verts),
                error=np.linalg.norm(verts_rec_cmh - verts),
                size=sizeof_fmt(Phi_cmh_sp.data.nbytes + Phi_cmh_sp.indptr.nbytes + Phi_cmh_sp.indices.nbytes),
                time=t_cmh,
            ))

    print "------------"
    print "TABLE START:"
    print "------------"
    print "mesh & basis & $K$ & $\\mu$ & error & size & time"
    for line in result_lines:
        print line

if __name__ == '__main__':
    main()
