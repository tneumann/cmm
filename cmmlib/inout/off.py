import numpy as np
from StringIO import StringIO

def save_mesh(filename, vertices=None, faces=None):
    if vertices is None:
        vertices = []
    if faces is None:
        faces = []
    with open(filename, 'w') as f:
        f.write("OFF\n%d %d 0\n" % (len(vertices), len(faces)))
        if len(vertices) > 1:
            np.savetxt(f, vertices, fmt="%f %f %f")
        if len(faces) > 1:
            for face in faces:
                fmt = " ".join(["%d"] * (len(face) + 1)) + "\n"
                f.write(fmt % ((len(face),) + tuple(map(int, face))))

def read_mesh(filename, no_colors=False):
    lines = open(filename).readlines()
    lines = [line for line in lines if line.strip() != '' and line[0] != '#']
    assert lines[0].strip() in ['OFF', 'COFF'], 'OFF header missing'
    has_colors = lines[0].strip() == 'COFF'
    n_verts, n_faces, _ = map(int, lines[1].split())
    vertex_data = np.loadtxt(
        StringIO(''.join(lines[2:2 + n_verts])), 
        dtype=np.float)
    if n_faces > 0:
        faces = np.loadtxt(StringIO(''.join(lines[2+n_verts:])), dtype=np.int)[:,1:]
    else:
        faces = None
    if has_colors:
        colors = vertex_data[:,3:].astype(np.uint8)
        vertex_data = vertex_data[:,:3]
    else:
        colors = None
    if no_colors:
        return vertex_data, faces
    else:
        return vertex_data, colors, faces
