import numpy as np
from scipy import sparse

from util import veclen, normalized
from laplacian import compute_mesh_laplacian
from prefactor import factorized


class GeodesicDistanceComputation(object):
    """
    Computation of geodesic distances on triangle meshes using the heat method from the impressive paper

        Geodesics in Heat: A New Approach to Computing Distance Based on Heat Flow
        Keenan Crane, Clarisse Weischedel, Max Wardetzky
        ACM Transactions on Graphics (SIGGRAPH 2013)

    Example usage:
        >>> compute_distance = GeodesicDistanceComputation(vertices, triangles)
        >>> distance_of_each_vertex_to_vertex_0 = compute_distance(0)

    """

    def __init__(self, verts, tris, m=1.0):
        self._verts = verts
        self._tris = tris
        # precompute some stuff needed later on
        e01 = verts[tris[:,1]] - verts[tris[:,0]]
        e12 = verts[tris[:,2]] - verts[tris[:,1]]
        e20 = verts[tris[:,0]] - verts[tris[:,2]]
        self._triangle_area = .5 * veclen(np.cross(e01, e12))
        unit_normal = normalized(np.cross(normalized(e01), normalized(e12)))
        self._un = unit_normal
        self._unit_normal_cross_e01 = np.cross(unit_normal, -e01)
        self._unit_normal_cross_e12 = np.cross(unit_normal, -e12)
        self._unit_normal_cross_e20 = np.cross(unit_normal, -e20)
        # parameters for heat method
        h = np.mean(map(veclen, [e01, e12, e20]))
        t = m * h ** 2
        # pre-factorize poisson systems
        Lc, vertex_area = compute_mesh_laplacian(verts, tris, area_type='lumped_mass')
        A = sparse.spdiags(vertex_area, 0, len(verts), len(verts))
        #self._factored_AtLc = splu((A - t * Lc).tocsc()).solve
        self._factored_AtLc = factorized((A - t * Lc).tocsc())
        #self._factored_L = splu(Lc.tocsc()).solve
        self._factored_L = factorized(Lc.tocsc())

    def __call__(self, idx):
        """
        computes geodesic distances to all vertices in the mesh
        idx can be either an integer (single vertex index) or a list of vertex indices
        or an array of bools of length n (with n the number of vertices in the mesh) 
        """
        u0 = np.zeros(len(self._verts))
        u0[idx] = 1.0
        # -- heat method, step 1
        u = self._factored_AtLc(u0).ravel()
        # -- heat method step 2
        # magnitude that we use to normalize the heat values across triangles
        n_u = 1. / (u[self._tris].sum(axis=1))
        # compute gradient
        grad_u =  self._unit_normal_cross_e01 * (n_u * u[self._tris[:,2]])[:,np.newaxis] \
                + self._unit_normal_cross_e12 * (n_u * u[self._tris[:,0]])[:,np.newaxis] \
                + self._unit_normal_cross_e20 * (n_u * u[self._tris[:,1]])[:,np.newaxis]
        X = - grad_u / veclen(grad_u)[:,np.newaxis]
        # -- heat method step 3
        # TODO this recomputes the cotangent weights, which is not at all necessary
        div_Xs = np.zeros(len(self._verts))
        for i1, i2, i3 in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]: # for edge i2 --> i3 facing vertex i1
            vi1, vi2, vi3 = self._tris[:,i1], self._tris[:,i2], self._tris[:,i3]
            e1 = self._verts[vi2] - self._verts[vi1]
            e2 = self._verts[vi3] - self._verts[vi1]
            e_opp = self._verts[vi3] - self._verts[vi2]
            #cot1 = 1 / np.tan(np.arccos( 
            #    (normalized(-e2) * normalized(-e_opp)).sum(axis=1)))
            cot1 = 0.5 * (e2 *  e_opp).sum(axis=1) / veclen(np.cross(e2, -e_opp))
            #cot2 = 1 / np.tan(np.arccos(
            #    (normalized(-e1) * normalized( e_opp)).sum(axis=1)))
            cot2 = 0.5 * (e1 * -e_opp).sum(axis=1) / veclen(np.cross(e1, +e_opp))
            div_Xs += np.bincount(
                vi1.astype(int),
                (((cot1[:,np.newaxis] * e1) * X).sum(axis=1) + ((cot2[:,np.newaxis] * e2) * X).sum(axis=1)),
                minlength=len(self._verts))
        # solve poisson system
        phi = self._factored_L(div_Xs).ravel()
        phi = phi - phi.min()
        phi = phi.max() - phi
        return phi
