import numpy as np
from mayavi import mlab
from tvtk.api import tvtk


def mesh_as_vtk_actor(verts, tris, compute_normals=True, return_polydata=True, scalars=None):
    pd = tvtk.PolyData(points=verts, polys=tris)
    if scalars is not None:
        pd.point_data.scalars = scalars
    if compute_normals:
        normals = tvtk.PolyDataNormals(input=pd, splitting=False).output
    else:
        normals = pd
    actor = tvtk.Actor(mapper=tvtk.PolyDataMapper(input=normals))
    if return_polydata:
        return actor, pd
    else:
        return actor

def vismesh(pts, tris, color=None, edge_visibility=False, shader=None, triangle_scalars=None, colors=None, **kwargs):
    if 'scalars' in kwargs and np.asarray(kwargs['scalars']).ndim == 2:
        colors = kwargs['scalars']
        del kwargs['scalars']
    tm = mlab.triangular_mesh(pts[:,0], pts[:,1], pts[:,2], tris, color=color, **kwargs)
    if shader is not None:
        tm.actor.property.load_material(shader)
        tm.actor.actor.property.shading = True
    diffuse = 1.0 if colors is not None else 0.8
    tm.actor.actor.property.set(
        edge_visibility=edge_visibility, line_width=1, 
        specular=0.0, specular_power=128., 
        diffuse=diffuse)
    if triangle_scalars is not None:
        tm.mlab_source.dataset.cell_data.scalars = triangle_scalars
        tm.actor.mapper.set(scalar_mode='use_cell_data', use_lookup_table_scalar_range=False,
                            scalar_visibility=True)
        if "vmin" in kwargs and "vmax" in kwargs:
            tm.actor.mapper.scalar_range = kwargs["vmin"], kwargs["vmax"]
    if colors is not None:
        # this basically is a hack which doesn't quite work, 
        # we have to completely replace the polydata behind the hands of mayavi
        tm.mlab_source.dataset.point_data.scalars = colors.astype(np.uint8)
        tm.actor.mapper.input = tvtk.PolyDataNormals(
            input=tm.mlab_source.dataset, splitting=False).output
    return tm

def compute_normals(pts, faces):
    pd = tvtk.PolyData(points=pts, polys=faces)
    n = tvtk.PolyDataNormals(input=pd, splitting=False)
    n.update()
    return n.output.point_data.normals.to_array()

def showmesh(pts, tris, **kwargs):
    mlab.clf()
    vismesh(pts, tris, **kwargs)
    if 'scalars' in kwargs:
        mlab.colorbar()
    mlab.show()

def viscroud(meshes, axis=0, padding=1.2, **kwargs):
    offset = np.zeros(3)
    offset[axis] = np.max([mesh[0][:,axis].ptp() for mesh in meshes]) * padding
    tms = []
    for i, mesh in enumerate(meshes):
        verts, tris = mesh[:2]
        scalars = mesh[2] if len(mesh) > 2 else None
        tm = vismesh(verts + offset * i, tris, scalars=scalars, **kwargs)
        tms.append(tm)
    return tms


