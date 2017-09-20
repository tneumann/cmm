try:
    from scikits.sparse.cholmod import cholesky
    factorized = lambda A: cholesky(A, mode='simplicial')
except ImportError:
    print "CHOLMOD not found - trying to use slower LU factorization from scipy"
    print "install scikits.sparse to use the faster cholesky factorization"
    from scipy.sparse.linalg import factorized

__all__ = ['factorized']

