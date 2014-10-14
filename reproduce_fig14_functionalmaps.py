import sys
from collections import defaultdict
from functools import partial
import numpy as np
from scipy.spatial import cKDTree
from joblib.memory import Memory
from joblib.parallel import Parallel, delayed, cpu_count

import matplotlib
matplotlib.use('WxAgg')
import pylab as pl

from cmmlib import cmm
from cmmlib.geodesic import GeodesicDistanceComputation
from cmmlib.inout import load_shape_pair
from cmmlib import align
from cmmlib.util import veclen
from cmmlib.vis.correspondence import show_correspondence
from cmmlib.vis.functional_map import plot_functional_map


mem = Memory(cachedir='cache', verbose=1)


def compute_geodesic_distances(verts, tris, sources):
    geo = GeodesicDistanceComputation(verts, tris, m=1.0)
    D = [geo(i).astype(np.float32) for i in sources]
    return np.array(D, dtype=np.float32)

@mem.cache
def compute_geodesic_distance_matrix(verts, tris):
    print "precomputing geodesic distance..."
    n_chunks = cpu_count()
    chunk_size = int(np.ceil(len(verts) / float(n_chunks)))
    sources = np.arange(len(verts))
    D = Parallel(n_chunks)(
        delayed(compute_geodesic_distances)(verts, tris, sources[i: i + chunk_size])
        for i in xrange(0, len(verts), chunk_size))
    return np.vstack(D)


def reorder_functional_map(C):
    Cnew = []
    Crows = C.tolist()
    while len(Crows) > 0:
        best_row = sorted(Crows, key=lambda row: np.abs(row[len(Cnew)]))[-1]
        Crows.remove(best_row)
        Cnew.append(best_row)
    return np.array(Cnew)

def main(experiment):
    swapped = False
    use_geodesic = True

    experiments = dict(
        cat = dict(
            file1 = 'meshes/tosca/cat0.mat',
            file2 = 'meshes/tosca/cat2.mat',
            mu = [0.7],
            Ks = np.arange(10, 50, 5),
        ),
        david = dict(
            file1 = 'meshes/tosca_holes/david0_halfed.obj',
            file2 = 'meshes/tosca_holes/david10_nolegs_nohead.obj',
            mu = [0.7],
            Ks = np.arange(10, 50, 5),
        ),
        hand = dict(
            file1 = 'meshes/hand_868.obj',
            file2 = 'meshes/hand_868_holes2.obj',
            mu = [5.0],
            Ks = np.arange(10, 50, 2),
        ),
    )
    if experiment not in experiments:
        print "Usage: python %s <experiment>" % sys.argv[0]
        print "where experiment is one of: %s" % (', '.join(experiments.keys()))
        sys.exit(1)

    exp = experiments[experiment]
    file1, file2 = exp['file1'], exp['file2']
    mu, Ks = exp['mu'], exp['Ks']

    print "this will take a while..."

    # load meshes
    verts1, tris1, verts2, tris2, ij = load_shape_pair(file1, file2, check=False, normalize=False)
    if swapped: # swapped?
        verts2, tris2, verts1, tris1 = verts1, tris1, verts2, tris2
        ij = ij[:,::-1]

    if use_geodesic:
        D = compute_geodesic_distance_matrix(verts1, tris1)

    eigenvector_algorithms = [
        ('MHB', partial(cmm.manifold_harmonics, scaled=False)),
    ]

    for cmu in mu:
        name = 'CMHB' if len(mu) == 1 else 'CMHB %f' % (cmu)
        eigenvector_algorithms.append(
            (name, partial(
                cmm.compressed_manifold_modes, 
                mu=cmu, maxiter=10000, verbose=0, scaled=False)))

    mapping_algorithms = [
        ('rotation', partial(align.optimal_rotation, allow_reflection=True)),
        #('permutation', partial(align.optimal_permutation, allow_reflection=True)),
    ]

    print "computing basis functions (first mesh)"
    runs = [(K, name, algo) for K in Ks for name, algo in eigenvector_algorithms]
    Phis1 = Parallel(-1)(
        delayed(algo)(verts1, tris1, K)
        for K, name, algo in runs)
    print "computing basis functions (second mesh)"
    Phis2 = Parallel(-1)(
        delayed(algo)(verts2, tris2, K)
        for K, name, algo in runs)

    # run for different number of eigenfunctions K
    errors_per_mapping_algorithm = defaultdict(lambda: defaultdict(list))
    C_by_mapping_algorithm = defaultdict(lambda: defaultdict(list))
    for (K, eigenvector_algorithm_name, algo), Phi1, Phi2 in zip(runs, Phis1, Phis2):
        for mapping_algorithm_name, mapping_algorithm in mapping_algorithms:
            # compute the mapping between the eigenfunctions
            C = mapping_algorithm(Phi1[ij[:,0]].T, Phi2[ij[:,1]].T)
            Phi1_rot = np.dot(C, Phi1.T).T

            # for all points in shape 2 where there is a correspondence to shape 1 (all j),
            # find corresponding point i in shape 1
            tree = cKDTree(Phi1_rot)
            distance_eigenspace, nearest = tree.query(Phi2[ij[:, 1]])
            ij_corr = np.column_stack((ij[:,0], nearest))

            # compute error
            if use_geodesic:
                errors = D[ij_corr[:,0], ij_corr[:,1]]
            else:
                errors = veclen(verts1[ij_corr[:, 0]] - verts1[ij_corr[:,1]])
            mean_error = np.mean(errors)
            errors_per_mapping_algorithm[mapping_algorithm_name][eigenvector_algorithm_name].append(mean_error)
            C_by_mapping_algorithm[mapping_algorithm_name][eigenvector_algorithm_name].append(C)
            print "%s/%s (K=%d) %.2f" % (mapping_algorithm_name, eigenvector_algorithm_name, K, mean_error)

    for mapping_algorithm_name, C_by_trial in C_by_mapping_algorithm.iteritems():
        for trial_name, Cs in C_by_trial.iteritems():
            # just use the last rotation matrix computed
            C = Cs[-2]
            pl.figure(mapping_algorithm_name + '_' + trial_name, figsize=(4,3.5))
            plot_functional_map(reorder_functional_map(C), newfig=False)
            pl.subplots_adjust(left=0.01, right=1, top=0.95, bottom=0.05)

    show_correspondence(
        verts1, tris1, verts2, tris2, ij, 
        points=5, color_only_correspondences=True,
        colormap='Paired', #offset_factor=(-1.8, +0.0, 0.),
        show_spheres=False, blend_factor=1.0, block=False)

    for mapping_algorithm_name, errors_per_trial in errors_per_mapping_algorithm.iteritems():
        #pl.rcParams.update({'font.size': 6})
        pl.figure(mapping_algorithm_name + '_' + file1 + '_' + file2, figsize=(2.5,2.5))
        pl.rc('text', usetex=False)
        style_by_name = dict(
            CMHB=dict(c=(6/255., 50/255., 100/255.), lw=2),
            MHB=dict(c=(121/255., 6/255., 34/255.), lw=1)
        )
        for trial_name, errors in sorted(errors_per_trial.items(), key=lambda (k,v): k):
            print trial_name, errors
            style = style_by_name.get(trial_name, {})
            pl.plot(Ks, errors, label=trial_name, **style)
        pl.locator_params(nbins=4)
        pl.legend().draggable()
        pl.autoscale(tight=True)
        pl.ylim(bottom=0)
        pl.grid()
        pl.xlabel(r'$K$')
        pl.ylabel(r'geodesic error')
        pl.tight_layout()
    pl.show()



if __name__ == '__main__':
    e = sys.argv[1] if len(sys.argv) > 1 else ""
    main(e)
