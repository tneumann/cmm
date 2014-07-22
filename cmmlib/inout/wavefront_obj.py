import numpy as np
import re

_triangle_regex = re.compile("^f\s+([^\/\s]+)/?\S*/?\S*\s+([^\/\s]+)/?\S*/?\S*\s+([^\/\s]+)", re.MULTILINE)
_normal_regex = re.compile("^vn\s+(\S+)\s+(\S+)\s+(\S+)", re.MULTILINE)
_vertex_regex = re.compile("^v\s+(\S+)\s+(\S+)\s+(\S+)", re.MULTILINE)

def loadobj(filename, load_normals=False):
    """ load a wavefront obj file
        loads vertices into a (x,y,z) struct array and vertex indices
        into a n x 3 index array 
        only loads obj files vertex positions and also
        only works with triangle meshes """
    vertices = np.fromregex(open(filename), _vertex_regex, np.float)
    if load_normals:
        normals = np.fromregex(open(filename), _normal_regex, np.float)
    triangles = np.fromregex(open(filename), _triangle_regex, np.int) - 1 # 1-based indexing in obj file format!
    if load_normals:
        return vertices, normals, triangles
    else:
        return vertices, triangles

def saveobj(filename, vertices, triangles, normals=None):
    with open(filename, 'w') as f:
        np.savetxt(f, vertices, fmt="v %f %f %f")
        if not normals is None:
            np.savetxt(f, normals, 
                       fmt="vn %f %f %f")
            if triangles is not None and len(triangles) > 0:
                np.savetxt(f, np.dstack((triangles, triangles)).reshape((-1, 6)) + 1, 
                           fmt="f %d//%d %d//%d %d//%d")
        else:
            if triangles is not None and len(triangles) > 0:
                np.savetxt(f, triangles+1, fmt="f " + ' '.join(['%d'] * triangles.shape[-1]))

