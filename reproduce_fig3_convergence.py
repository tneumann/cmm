import sys
from collections import defaultdict, namedtuple
import numpy as np

from cmmlib.inout import load_mesh
from cmmlib import cmm
from cmmlib.vis.weights import show_weights



class Logger(object):
    def __init__(self):
        self.r_primals = []
        self.r_duals = []

    def __call__(self, *args, **kwargs):
        self.r_primals.append(kwargs['r_primal'])
        self.r_duals.append(kwargs['r_dual'])


def test_convergence(verts, tris, K, mu, Phi_init, algorithm, maxiter, **kwargs):
    logger = Logger()
    Phi, info = cmm.compressed_manifold_modes(
        verts, tris, K, mu, algorithm=algorithm,
        callback=logger,
        init=Phi_init,
        tol_rel=0, tol_abs=0,
        scaled=False, order=False, maxiter=maxiter, return_info=True, 
        verbose=0, **kwargs)
    return Phi, info, logger


def main():
    # prepare data
    verts, tris = load_mesh('meshes/bumpy_cube6.obj')
    K = 8 + 6 + 12
    mu = 20.0
    maxiter = 500
    num_experiments = 1

    # prepare data structures
    ResultData = namedtuple('ResultData', 'Phi info logger')
    data_per_algo = defaultdict(list)
    # parameters passed to algorithms - to be fair,
    # turn off automatic penalty adjustment in our method
    # set check_interval = 1 to force callback to be called in every iteration
    algorithms = [
        ('osher', cmm.solve_compressed_osher,
            dict()),
        ('ours', cmm.solve_compressed_splitorth,
            dict(auto_adjust_penalty=False, check_interval=1)),
    ]

    # run the algorithm with same input mesh and
    # different random initializations
    print "will now run the method with %d different initializations" % num_experiments
    print "the primal residual with our method should consistently be lower compared to Oshers et al."
    for random_seed in range(num_experiments):
        np.random.seed(random_seed)
        Phi_init = np.random.random((len(verts), K))
        print "--"
        print "running experiment %d" % random_seed
        for name, method, options in algorithms:
            Phi, info, logger = test_convergence(
                verts, tris, K, mu,
                Phi_init, method, maxiter, **options)
            print name, info['r_primal']
            data_per_algo[name].append(ResultData(Phi, info, logger))

    # visualize results
    import matplotlib
    if len(sys.argv) > 1:
        matplotlib.use('pgf') # TODO PGF
    else:
        matplotlib.use('WxAgg')

    import pylab as pl
    pl.rc('text', usetex=True)
    pl.rc('font', size=6)
    pl.figure(figsize=(0.9, 1.1))
    plot_fun = pl.semilogy
    plot_fun(data_per_algo['osher'][0].logger.r_primals, c='r',
             label=r'osher')
    plot_fun(data_per_algo['ours'][0].logger.r_primals, c='g',
             label=r'ours')
    pl.legend(loc='center right').draggable()
    pl.locator_params(nbins=4, axis='x')
    pl.xlabel('iteration')
    pl.ylabel(r'$  \left\Vert r \right\Vert_2$')
    pl.gca().yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=4))

    if len(sys.argv) > 2:
        pl.savefig(sys.argv[1])  #, bbox_inches='tight')
    else:
        pl.ion()
        pl.show()

    # let us first visualize the modes where things go wrong
    # which are most probably those where the primal residual is very high
    Q = data_per_algo['osher'][0].info['Q']
    P = data_per_algo['osher'][0].info['P']
    worst_first = ((Q - P)**2).sum(axis=0).argsort()[::-1]

    show_weights(
        verts, tris,
        (data_per_algo['osher'][0].Phi[:, worst_first], 
         data_per_algo['ours'][0].Phi[:, worst_first]), 
        ('osher', 'ours')
    )

if __name__ == '__main__':
    main()
