from itertools import count
import logging
import numpy as np
from scipy import sparse
from scikits.sparse.cholmod import cholesky, CholmodError
from scipy.sparse.linalg import eigsh

from laplacian import compute_mesh_laplacian
from orthomax import orthomax



def compressed_manifold_modes(verts, tris, K, mu, init=None, scaled=False,
                              return_info=False, return_eigenvalues=False, return_D=False,
                              order=True, **algorithm_params):
    Q, vertex_area = compute_mesh_laplacian(verts, tris, 'cotangent', 
                                            return_vertex_area=True, area_type='lumped_mass')
    if scaled:
        D = sparse.spdiags(np.sqrt(vertex_area), 0, len(verts), len(verts))
        Dinv = sparse.spdiags(1 / np.sqrt(vertex_area), 0, len(verts), len(verts))
    else:
        D = Dinv = None

    if init == 'mh':
        Phi_init = manifold_harmonics(verts, tris, K)
    elif init == 'varimax':
        Phi_init = varimax_modes(verts, tris, K)
    elif type(init) == np.ndarray:
        Phi_init = init
    else:
        Phi_init = None

    Phi, info = solve_compressed_splitorth(
        Q, K, mu1=mu, Phi_init=Phi_init, D=D, Dinv=Dinv, 
        **algorithm_params)

    l = eigenvalues_from_eigenvectors(verts, tris, Phi, Q=Q)
    if order:
        idx = np.abs(l).argsort()
        Phi = Phi[:, idx]
        l = l[idx]

    result = [Phi]
    if return_eigenvalues:
        result.append(l)
    if return_D:
        if D is not None:
            result.append(D * D)
        else:
            result.append(None)
    if return_info:
        result.append(info)
    if len(result) == 1:
        return result[0]
    else:
        return result


def manifold_harmonics(verts, tris, K, scaled=True, return_D=False, return_eigenvalues=False):
    Q, vertex_area = compute_mesh_laplacian(
        verts, tris, 'cotangent', 
        return_vertex_area=True, area_type='lumped_mass'
    )
    if scaled:
        D = sparse.spdiags(vertex_area, 0, len(verts), len(verts))
    else:
        D = sparse.spdiags(np.ones_like(vertex_area), 0, len(verts), len(verts))

    try:
        lambda_dense, Phi_dense = eigsh(-Q, M=D, k=K, sigma=0)
    except RuntimeError, e:
        if e.message == 'Factor is exactly singular':
            logging.warn("factor is singular, trying some regularization and cholmod")
            chol_solve = cholesky(-Q + sparse.eye(Q.shape[0]) * 1.e-9, mode='simplicial')
            OPinv = sparse.linalg.LinearOperator(Q.shape, matvec=chol_solve)
            lambda_dense, Phi_dense = eigsh(-Q, M=D, k=K, sigma=0, OPinv=OPinv)
        else:
            raise e

    result = [Phi_dense]
    if return_D:
        result.append(D)
    if return_eigenvalues:
        result.append(lambda_dense)
    if len(result) == 1:
        return result[0]
    else:
        return result


def varimax_modes(verts, tris, K, scaled=False):
    Phi_dense = manifold_harmonics(verts, tris, K, scaled=scaled)
    return orthomax(Phi_dense, maxiter=100, gamma=1.0)


def eigenvalues_from_eigenvectors(verts, tris, eigs, Q=None):
    if Q is None:
        Q = compute_mesh_laplacian(verts, tris, 'cotangent', return_vertex_area=False)
    return (eigs * (-Q * eigs)).sum(axis=0)


def shrink(u, delta):
    """ proximal operator of L1 """
    return np.multiply(np.sign(u), np.maximum(0, np.abs(u) - delta))


def solve_compressed_splitorth(L, K, mu1=10., Phi_init=None, maxiter=None, callback=None, 
                               D=None, Dinv=None,
                               rho=1.0, auto_adjust_penalty=True, 
                               rho_adjust=2.0, rho_adjust_sensitivity=5.,
                               tol_abs=1.e-8, tol_rel=1.e-6,
                               verbose=100, check_interval=10):
    N = L.shape[0]  # alias
    mu = (mu1 / float(N))

    # initial variables
    if Phi_init is None:
        Phi = np.linalg.qr(np.random.uniform(-1, 1, (L.shape[0], K)))[0]
    else:
        Phi = Phi_init
    P = Q = Phi
    U = np.zeros((2, Phi.shape[0], Phi.shape[1]))

    # status variables for Phi-solve
    Hsolve = None
    refactorize = False

    # iteration state
    iters = count() if maxiter is None else xrange(maxiter)
    converged = False

    info = {}

    if D is None:
        D = sparse.identity(N)
        Dinv = sparse.identity(N)

    for i in iters:
        # update Phi
        _PA = D * (P - U[1] + Q - U[0])
        _PP = np.dot(_PA.T, _PA)
        S, sigma, St = np.linalg.svd(_PP)
        Sigma_inv = np.diag(np.sqrt(1.0 / sigma))
        Phi = Dinv * (_PA.dot(S.dot(Sigma_inv.dot(St))))

        # update Q
        Q_old = Q
        Q = shrink(Phi + U[0], mu / rho)

        # update P
        if Hsolve is None or refactorize:
            A = (-L - L.T + sparse.eye(L.shape[0], L.shape[0]) * rho).tocsc()
            if Hsolve is None:
                Hsolve = cholesky(A, mode='simplicial')
            elif refactorize:
                Hsolve.cholesky_inplace(A)
                refactorize = False
        rhs = np.asfortranarray(rho * (Phi + U[1]))
        P_old = P
        P = Hsolve(rhs)

        # update residuals
        U[0] += Phi - Q
        U[1] += Phi - P

        if i % check_interval == 0:
            # compute primal and dual residual
            snorm = np.sqrt((rho * ((Q - Q_old)**2 + (P - P_old)**2)).sum())
            rnorm = np.sqrt(((Phi - Q)**2 + (Phi - P)**2).sum())
            if auto_adjust_penalty:
                if rnorm > rho_adjust_sensitivity * snorm:
                    rho *= rho_adjust
                    U /= rho_adjust
                    refactorize = True
                elif snorm > rho_adjust_sensitivity * rnorm:
                    rho /= rho_adjust
                    U *= rho_adjust
                    refactorize = True

            if callback is not None:
                try:
                    callback(L, mu, Phi, P, Q, r_primal=rnorm, r_dual=snorm)
                except StopIteration:
                    converged = True

            # convergence checks
            eps_pri = np.sqrt(Phi.shape[1]*2) * tol_abs + tol_rel * max(map(np.linalg.norm, [Phi, np.vstack((Q, P))]))
            eps_dual = np.sqrt(Phi.shape[1]) * tol_abs + tol_rel * np.linalg.norm(rho * U)
            # TODO check if convergence check is correct for 3-function ADMM
            if rnorm < eps_pri and snorm < eps_dual or converged:
                print "converge!"
                converged = True
            if verbose and (i % verbose == 0 or converged or (maxiter is not None and i == maxiter - 1)):
                sparsity = np.sum(mu * np.abs(Q))
                eig = (Phi * (L * Phi)).sum()
                gap1 = np.linalg.norm(Q - Phi)
                gap2 = np.linalg.norm(P - Phi)
                #ortho = np.linalg.norm(Phi.T.dot((D.T * D) * Phi) - np.eye(Phi.shape[1]))
                print i, 
                print "o %0.8f" % (sparsity + eig), # %0.4f %0.4f" % (sparsity + eig, sparsity, eig), 
                print " | r [%.8f %.8f %.8f] s [%.8f]" % (gap1, gap2, rnorm, snorm)
            if converged:
                break

    info['num_iters'] = i
    info['r_primal'] = rnorm
    info['r_dual'] = snorm
    info['Q'] = Q
    info['P'] = P
    info['Phi'] = Phi
    return Phi, info
