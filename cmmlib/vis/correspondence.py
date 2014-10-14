import numpy as np
from mayavi import mlab

from mesh import vismesh
from ..geodesic import GeodesicDistanceComputation
from ..util import veclen


def compute_fake_weights(verts, tris, ci, expo=7):
    geo = GeodesicDistanceComputation(verts, tris)
    H = np.column_stack([geo(i) for i in ci])
    H = H.max(axis=0) - H
    H = H / H.max(axis=0)
    H **= expo
    H /= H.sum(axis=1)[:,np.newaxis]
    return H


def show_correspondence(verts1, tris1, verts2, tris2, ij, points=5, show_spheres=True, 
                        scalars=None, colormap='gist_rainbow', blend_factor=0.9, 
                        compute_blend_weights=compute_fake_weights,
                        color_only_correspondences=1, color_no_correspondence=(0,0,0),
                        offset_factor=(1.5, 0., 0.), block=True,
                       ):
    mlab.figure(bgcolor=(1,1,1))
    # select sparse points to visualize
    if type(points) is int:
        i_sel = [0]
        geo = GeodesicDistanceComputation(verts1, tris1)
        for n in xrange(points-1):
            d = geo(ij[i_sel,0])[ij[:,0]]
            i_sel.append(d.argmax())
    else:
        i_sel = points
    #i_sel = np.random.randint(0, len(ij), 10)
    ij_sel = np.column_stack((ij[i_sel,0], ij[i_sel, 1]))
    # color per marker - value between 0 and 1 which is passed through a color map later on
    color = np.linspace(0, 1, len(ij_sel))
    # prepare visualization
    offset = verts2.ptp(axis=0) * offset_factor#(verts2[:,0].ptp() * offset_factor, 0, 0)
    p1 = verts1[ij_sel[:,0]]
    p2 = verts2[ij_sel[:,1]] + offset
    v = p2 - p1
    # visualize!
    # correspondence arrows
    quiv = mlab.quiver3d(p1[:,0], p1[:,1], p1[:,2], v[:,0], v[:,1], v[:,2],
                         scale_factor=1, line_width=2, mode='2ddash', scale_mode='vector',
                         scalars=color, colormap=colormap)
    quiv.glyph.color_mode = 'color_by_scalar'
    # show sparse points as spheres
    if show_spheres:
        h = veclen(verts1[tris1[:,0]] - verts1[tris1[:,1]]).mean() * 2
        mlab.points3d(p1[:,0], p1[:,1], p1[:,2], color, scale_mode='none', 
                      scale_factor=h, colormap=colormap, resolution=32)
        mlab.points3d(p2[:,0], p2[:,1], p2[:,2], color, scale_mode='none', 
                      scale_factor=h, colormap=colormap, resolution=32)
    # make colors for the meshes
    if scalars is None:
        H = compute_blend_weights(verts1, tris1, ij_sel[:,0])
        # interpolate colors
        lut = quiv.module_manager.scalar_lut_manager.lut.table.to_array()
        lut_colors_rgb = lut[(color * (lut.shape[0]-1)).astype(np.int), :3].astype(np.float)
        scalars = ((1 - blend_factor) * lut_colors_rgb[H.argmax(axis=1)] + \
                   blend_factor * (H[:,:,np.newaxis] * lut_colors_rgb[np.newaxis,:,:]).sum(1))
        scalars = np.uint8(scalars)
        if color_only_correspondences:
            scalars_filtered = np.zeros((len(verts1), 3))
            scalars_filtered[:] = np.array(color_no_correspondence)[np.newaxis] * 255
            scalars_filtered[ij[:,0]] = scalars[ij[:,0]]
            scalars = scalars_filtered
    scalars2 = np.zeros((len(verts2), 3))
    scalars2[:] = np.array(color_no_correspondence)[np.newaxis] * 255
    scalars2[ij[:,1]] = scalars[ij[:,0]]
    # show meshes
    vismesh(verts1, tris1, scalars=scalars)
    vismesh(verts2 + offset, tris2, scalars=scalars2)
    if block:
        mlab.show()
