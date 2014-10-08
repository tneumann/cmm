from os import path
import logging
import numpy as np
from itertools import cycle, product
from mayavi import mlab
from traits.api import Any, HasTraits, Range, Int, String, Instance, on_trait_change, Bool, List, Button
from tvtk.pyface.scene_editor import SceneEditor
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene
from traitsui.api import View, Item, HGroup, Group, EnumEditor
from pyface.api import DirectoryDialog, OK

def _centered(x):
    if x is None:
        return None
    vmax = np.maximum(np.abs(x.max(0)), np.abs(x.min(0)))
    vmin = -vmax
    x = ((x - vmin) / (vmax - vmin))
    return x

class WeightsVisualization(HasTraits):
    scene = Instance(MlabSceneModel, (), kw=dict(background=(1,1,1)))
    weight_index = Range(value=0, low='_min_weight', high='_max_weight')
    display_all = Bool(True)
    _min_weight = Int(0)
    _max_weight = Int()
    selected_mesh = String()
    _names = List(String)

    save_all = Button()

    def __init__(self, meshes, center=True, vmin=None, vmax=None, offset_axis=0, offset_spacing=0.2, contours=None, colormap='RdBu', actor_options=dict(), **kwargs):
        HasTraits.__init__(self)

        if type(meshes) is dict:
            names, verts, tris, weights = zip(*[(n, v, t, w) for n, (v, t, w) in meshes.iteritems()])
        elif type(meshes) in [list, tuple]:
            names, verts, tris, weights = [], [], [], []
            for i, mesh in enumerate(meshes):
                # (verts, tris, weights, name)
                verts.append(mesh[0])
                tris.append(mesh[1])
                weights.append(mesh[2])
                if len(mesh) < 4:
                    names.append(str(i))
                else:
                    names.append(mesh[3])
        else:
            raise ValueError, 'illegal value for parameter "meshes"'

        ## single arrays given?
        #share_verts, share_tris = False, False
        #if type(weights) is np.ndarray and weights.ndim == 2:
        #    weights = [weights]
        #if type(verts) is np.ndarray and verts.ndim == 2:
        #    verts = cycle([verts])
        #    share_verts = True
        #if type(tris) is np.ndarray and tris.ndim == 2:
        #    tris = cycle([tris])
        #    share_tris = True
        #else:
        #    if not share_tris and len(tris) != len(weights):
        #        raise ValueError, "need the same number of weight and triangle arrays"
        #    if not share_verts and len(verts) != len(weights):
        #        raise ValueError, "need the same number of weight and vertex arrays"
        #    if names is not None and len(names) != len(weights):
        #        raise ValueError, "need the same number of weight arrays and names"

        if names is not None:
            assert len(set(names)) == len(names)

        self._weights = map(_centered, weights) if center else weights
        self._verts = verts
        self._tris = tris
        self._max_weight = max(w.shape[1] - 1 for w in self._weights if w is not None)
        self._names = names
        #self._names = map(str, range(len(self._weights))) if names is None else names
        #if len(weights) == 2:
        #    self.display_all = True

        # visualize each mesh
        self._trimeshes = []
        self._offsets = []
        offset = np.zeros(3)
        for i, (verts_i, tris_i, weights_i) in enumerate(zip(verts, tris, weights)):
            if i > 0:
                offset[offset_axis] += verts_i[:,offset_axis].ptp() * (1 + offset_spacing)
            if center:
                vmin, vmax = 0, 1
            else:
                vmin, vmax = vmin, vmax
            # draw mesh
            tm = mlab.triangular_mesh(
                verts_i[:,0], verts_i[:,1], verts_i[:,2], tris_i, 
                scalars=weights_i[:,0] if weights_i is not None else None, 
                colormap=colormap,
                vmin=vmin, vmax=vmax,
                figure=self.scene.mayavi_scene)
            if weights_i is None:
                tm.actor.mapper.scalar_visibility = False
            # disable normal splitting
            tm.parent.parent.filter.splitting = False

            if contours is not None and weights_i is not None:
                from mayavi.modules.surface import Surface
                engine = tm.scene.engine
                tm_contour = Surface()
                engine.add_filter(tm_contour, tm.module_manager)
                tm_contour.enable_contours = True
                tm_contour.contour.number_of_contours = contours
                tm_contour.actor.mapper.scalar_visibility = False
                tm_contour.actor.property.set(color = (0.0, 0.0, 0.0), line_width=4)

                self._trimeshes.append([tm, tm_contour])
            else:
                self._trimeshes.append([tm])


            #tm.actor.actor.property.edit_traits()
            # for the next mesh, add the extent of the current mesh (plus spacing) to the offset
            self._offsets.append(offset.copy())
            tm.actor.property.set(**actor_options)
            #if handle_points is not None:
            #    h = verts2.ptp()
            #    mlab.points3d(handle_points[:,0], handle_points[:,1], handle_points[:,2], scale_factor=h / 20)

        #actor_prop.edit_traits()
        self.selected_mesh = self._names[0]
        self._update_view()
        self._reposition_meshes()

    @on_trait_change('weight_index')
    def _update_view(self):
        for tms, w in zip(self._trimeshes, self._weights):
            tm = tms[0]
            if w is None:
                continue
            try:
                tm.mlab_source.set(
                    scalars=w[:,self.weight_index])
            except IndexError:
                logging.warn("coult not reference index %d" % self.weight_index)

    @on_trait_change('display_all')
    def _reposition_meshes(self):
        if self.display_all:
            for tms, offset in zip(self._trimeshes, self._offsets):
                for tm in tms:
                    tm.actor.actor.position = offset
        else:
            for tms in self._trimeshes:
                for tm in tms:
                    tm.actor.actor.position = (0, 0, 0)

    @on_trait_change('display_all, selected_mesh')
    def _update_mesh_visibilities(self):
        for name, tms in zip(self._names, self._trimeshes):
            for tm in tms:
                tm.visible = self.display_all or name == self.selected_mesh

    def _save_all_fired(self):
        file_dialog = DirectoryDialog(action = 'open', title = 'Select Directory')
        if file_dialog.open() == OK:
            out_path = file_dialog.path
            if self.display_all:
                items = [(None, k) for k in xrange(self._max_weight + 1)]
            else:
                items = product(self._names, xrange(self._max_weight + 1))
            for name, k in items:
                if name is not None:
                    self.selected_mesh = name
                self.weight_index = k
                mlab.savefig(path.join(out_path, "%s_%03d.png" % (name, k)))

    view = View(
        Group(
        HGroup(
            Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                 height=400, width=400, show_label=False),
        ),
        HGroup(
            Item('weight_index', label='idx'),
            Item('display_all', label='all'),
            Item('selected_mesh', label='which', 
                 visible_when="len(object._names) > 1 and not object.display_all",
                 editor=EnumEditor(
                     name='object._names'),
                ),
            Item('save_all'),
        ),
        ),
        resizable=True, title="Mesh Weights",
    )

def show_many_weights(meshes, **kwargs):
    WeightsVisualization(meshes, **kwargs).configure_traits()

def show_weights(verts, tris, multi_weights, names=None, **kwargs):
    if type(multi_weights) is np.ndarray and multi_weights.ndim == 2:
        multi_weights = [multi_weights] # single weight vector
    if names is None:
        meshes = [[verts, tris, w] for w in multi_weights]
    else:
        meshes = [[verts, tris, w, n] for w, n in zip(multi_weights, names)]
    show_many_weights(meshes, **kwargs)

