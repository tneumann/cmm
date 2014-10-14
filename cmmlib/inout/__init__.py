from os import path
import numpy as np
import h5py
from scipy.io import loadmat
from scipy.sparse import csgraph
from scipy import sparse

from wavefront_obj import loadobj
import off
from ..util import filter_reindex


def load_nrw_mat(mat_file):
    data = loadmat(mat_file)
    verts = np.column_stack((
        data['surface'][c][0,0].ravel()
        for c in 'XYZ'))
    faces = data['surface']['TRIV'][0,0].astype(np.int) - 1 # 1 based indexing in mat files
    return verts, faces

def load_mesh(filename, check=True, normalize=True, *args, **kwargs):
    ext = path.splitext(filename)[1].lower()
    if ext == '.obj':
        result = loadobj(filename, *args, **kwargs)
    elif ext == '.off':
        if not 'no_colors' in kwargs:
            kwargs['no_colors'] = True
        result = off.read_mesh(filename, *args, **kwargs)
        if len(result) == 3:
            result = [result[0], result[2], result[1]]
    elif ext == '.mat':
        result = load_nrw_mat(filename, *args, **kwargs)
    elif ext == '.ply':
        from tvtk.api import tvtk
        reader = tvtk.PLYReader(file_name=filename)
        reader.update()
        pd = reader.output
        faces = pd.polys.to_array().reshape((-1, 4))
        assert np.all(faces[:,0] == 3) # ensure triangles
        result = pd.points.to_array(), faces[:,1:]
    else:
        raise ValueError, "cannot load meshes with extension %s" % ext
    if normalize:
        verts = result[0]
        verts[:] = verts - verts.mean(axis=0)
        verts[:] = verts / verts.std()
    if check:
        check_mesh(result[0], result[1], filename)
    return result

def save_coff(filename, vertices, tris, colors=None):
    with open(filename, 'w') as f:
        header = 'OFF' if colors is None else 'COFF'
        f.write("%s\n%d %d 0\n" % (header, len(vertices), len(tris)))
        if colors is None:
            np.savetxt(f, vertices, fmt="%f %f %f")
        else:
            np.savetxt(f, np.hstack((vertices, colors)), fmt="%f %f %f %d %d %d")
        np.savetxt(f, np.insert(tris, 0, 3, axis=1), fmt="%d")

def check_mesh(verts, tris, filename=None):
    ij = np.r_[np.c_[tris[:,0], tris[:,1]], 
               np.c_[tris[:,0], tris[:,2]], 
               np.c_[tris[:,1], tris[:,2]]]
    G = sparse.csr_matrix((np.ones(len(ij)), ij.T), shape=(verts.shape[0], verts.shape[0]))
    n_components, labels = csgraph.connected_components(G, directed=False)
    if n_components > 1:
        size_components = np.bincount(labels)
        if len(size_components) > 1:
            raise ValueError, "found %d connected components in the mesh (%s)" % (n_components, filename)
        keep_vert = labels == size_components.argmax()
    else:
        keep_vert = np.ones(verts.shape[0], np.bool)
    verts = verts[keep_vert, :]
    tris = filter_reindex(keep_vert, tris[keep_vert[tris].all(axis=1)])
    return verts, tris

def save_eigens(filename, verts, tris, Phi):
    with h5py.File(filename, 'w') as f:
        f['verts'] = verts
        f['tris'] = tris
        f['Phi'] = Phi

def load_eigens(filename):
    with h5py.File(filename, 'r') as f:
        return f['verts'].value, f['tris'].value, f['Phi'].value

def save_eigenpair(filename, verts1, tris1, Phi1, verts2, tris2, Phi2, ij):
    with h5py.File(filename, 'w') as f:
        f['verts1'] = verts1
        f['tris1'] = tris1
        f['Phi1'] = Phi1
        f['verts2'] = verts2
        f['tris2'] = tris2
        f['Phi2'] = Phi2
        f['ij'] = ij

def load_eigenpair(filename):
    with h5py.File(filename, 'r') as f:
        return [f[name].value for name in \
                ['verts1', 'tris1', 'Phi1', 'verts2', 'tris2', 'Phi2', 'ij']]

def load_shape_pair(filename1, filename2, **kwargs):
    basedir1 = path.dirname(filename1)
    basedir2 = path.dirname(filename2)
    basename1 = path.splitext(path.basename(filename1))[0]
    basename2 = path.splitext(path.basename(filename2))[0]
    verts1, tris1 = load_mesh(filename1, **kwargs)
    verts2, tris2 = load_mesh(filename2, **kwargs)
    if np.all(tris1 == tris2):
        print "same topo"
        ij = np.column_stack((np.arange(len(verts1)), np.arange(len(verts2))))
    else:
        map_file = path.join(basedir1, "%s_%s.ij" % (basename1, basename2))
        ij = np.loadtxt(map_file, dtype=np.int)
    return verts1, tris1, verts2, tris2, ij
