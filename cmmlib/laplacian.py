import numpy as np
from scipy import sparse

from util import veclen


def compute_mesh_laplacian(verts, tris, weight_type='cotangent',
                           return_vertex_area=True, area_type='mixed',
                           add_diagonal=True):
    """
    computes a sparse matrix representing the 
    discretized laplace-beltrami operator of the mesh
    given by n vertex positions ("verts") and a m triangles ("tris")

    verts: (n, 3) array (float)
    tris: (m, 3) array (int) - indices into the verts array
    weight_type: either 'mean_value', 'uniform' or 'cotangent' (default)
    return_vertex_area: wether to return another area with the areas per vertex
    area_type: can be 'mixed' or 'lumped_mass'

    if weight_type == 'cotangent':
        computes the conformal weights ("cotangent weights") for the mesh, ie:
        w_ij = - .5 * (cot \alpha + cot \beta)

    if weight_type == 'mean_value':
        computes mean value coordinates for the mesh
        w_ij = - (tan(theta1_ij / 2) + tan(theta2_ij / 2)) / || v_i - v_j ||

    if weight_type == 'uniform':
        w_ij = - 1

    for all weight types:
        w_ii = sum(w_ij for j in [1..n])


    if area_type == 'mixed':
        compute the vertex area as the voronoi area for non-obtuse triangles,
        use the barycentric area for obtuse triangles 
        (according to Mark Meyer's 2002 paper)

    if area_type == 'lumped_mass':
        compute vertex area by equally dividing the triangle area to the vertices,
        e.g. area of vertex i is the sum of areas of adjacent triangles divided by 3

    See:
        Olga Sorkine, "Laplacian Mesh Processing"
        and also
        Mark Meyer et al., "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds"
        and for theoretical comparison of different discretizations, see
        Max Wardetzky et al., "Discrete Laplace operators: No free lunch"

    returns matrix L that computes the laplacian coordinates, e.g. L * x = delta
    """
    if area_type not in ['mixed', 'lumped_mass']:
        raise ValueError('unknown area type: %s' % area_type)
    if weight_type not in ['cotangent', 'mean_value', 'uniform']:
        raise ValueError('unknown weight type: %s' % weight_type)

    n = len(verts)
    # we consider the triangle P, Q, R
    iP = tris[:, 0]
    iQ = tris[:, 1]
    iR = tris[:, 2]
    # edges forming the triangle
    PQ = verts[iP] - verts[iQ] # P--Q
    QR = verts[iQ] - verts[iR] # Q--R
    RP = verts[iR] - verts[iP] # R--P
    if weight_type == 'cotangent' or (return_vertex_area and area_type == 'mixed'):
        # compute cotangent at all 3 points in triangle PQR
        cotP = -1 * (PQ * RP).sum(axis=1) / veclen(np.cross(PQ, RP)) # angle at vertex P
        cotQ = -1 * (QR * PQ).sum(axis=1) / veclen(np.cross(QR, PQ)) # angle at vertex Q
        cotR = -1 * (RP * QR).sum(axis=1) / veclen(np.cross(RP, QR)) # angle at vertex R

    # compute weights and indices
    if weight_type == 'cotangent':
        I =       np.concatenate((  iP,   iR,    iP,   iQ,    iQ,   iR))
        J =       np.concatenate((  iR,   iP,    iQ,   iP,    iR,   iQ))
        W = 0.5 * np.concatenate((cotQ, cotQ,  cotR, cotR,  cotP, cotP))

    elif weight_type == 'mean_value':
        # TODO: I didn't check this code yet
        PQlen = 1 / veclen(PQ)
        QRlen = 1 / veclen(QR)
        RPlen = 1 / veclen(RP)
        PQn = PQ * PQlen[:,np.newaxis] # normalized
        QRn = QR * QRlen[:,np.newaxis]
        RPn = RP * RPlen[:,np.newaxis]
        # TODO pretty sure there is a simpler solution to those 3 formulas
        tP = np.tan(0.5 * np.arccos((PQn * -RPn).sum(axis=1)))
        tQ = np.tan(0.5 * np.arccos((-PQn * QRn).sum(axis=1)))
        tR = np.tan(0.5 * np.arccos((RPn * -QRn).sum(axis=1)))
        I = np.concatenate((      iP,       iP,       iQ,       iQ,       iR,       iR))
        J = np.concatenate((      iQ,       iR,       iP,       iR,       iP,       iQ))
        W = np.concatenate((tP*PQlen, tP*RPlen, tQ*PQlen, tQ*QRlen, tR*RPlen, tR*QRlen))

    elif weight_type == 'uniform':
        # this might add an edge twice to the matrix
        # but prevents the problem of boundary edges going only in one direction
        # we fix this problem after the matrix L is constructed
        I = np.concatenate((iP, iQ,  iQ, iR,  iR, iP))
        J = np.concatenate((iQ, iP,  iR, iQ,  iP, iR))
        W = np.ones(len(tris) * 6)

    # construct sparse matrix
    # notice that this will also sum duplicate entries of (i,j), 
    # which is explicitely assumed by the code above
    L = sparse.csr_matrix((W, (I, J)), shape=(n, n))
    if weight_type == 'uniform':
        # because we probably add weights in both directions of an edge earlier, 
        # and the csr_matrix constructor sums them, some values in L might be 2 instead of 1
        # so reset them
        L.data[:] = 1
    # add diagonal entries as the sum across rows
    if add_diagonal:
        L = L - sparse.spdiags(L * np.ones(n), 0, n, n)

    if return_vertex_area:
        if area_type == 'mixed':
            # compute voronoi cell areas
            aP = 1/8. * (cotR * (PQ**2).sum(axis=1) + cotQ * (RP**2).sum(axis=1)) # area at point P
            aQ = 1/8. * (cotR * (PQ**2).sum(axis=1) + cotP * (QR**2).sum(axis=1)) # area at point Q
            aR = 1/8. * (cotQ * (RP**2).sum(axis=1) + cotP * (QR**2).sum(axis=1)) # area at point R
            # replace by barycentric areas for obtuse triangles
            triangle_area = .5 * veclen(np.cross(PQ, RP))
            for i, c in enumerate([cotP, cotQ, cotR]):
                is_x_obtuse = c < 0 # obtuse at point?
                # TODO: the paper by Desbrun says that we should divide by 1/2 or 1/4,
                #       but according to other code I found we should divide by 1 or 1/2
                #       check which scheme is correct!
                aP[is_x_obtuse] = triangle_area[is_x_obtuse] * (1 if i == 0 else 1/2.)
                aQ[is_x_obtuse] = triangle_area[is_x_obtuse] * (1 if i == 1 else 1/2.)
                aR[is_x_obtuse] = triangle_area[is_x_obtuse] * (1 if i == 2 else 1/2.)
            area = np.bincount(iP, aP, minlength=n) + \
                    np.bincount(iQ, aQ, minlength=n) + np.bincount(iR, aR, minlength=n)

        elif area_type == 'lumped_mass':
            lump_area = veclen(np.cross(PQ, RP)) / 6.
            area = sum(np.bincount(tris[:,i], lump_area, minlength=n) for i in xrange(3))

        return L, area
    else:
        return L

