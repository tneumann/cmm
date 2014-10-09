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
    display_labels = Bool(True)
    _min_weight = Int(0)
    _max_weight = Int()
    selected_mesh = String()
    _names = List(String)

    save_all = Button()

    def __init__(self, meshes, center=True, vmin=None, vmax=None, offset_axis=0, offset_spacing=0.2, contours=None, colormap='RdBu', show_labels=False, actor_options=dict(), offset_axis2=1, offset_spacing2=0.5, label_offset_axis=None, label_offset=1.1, num_columns=None, **kwargs):
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
        valid_weights = [w.shape[-1] - 1 for w in self._weights if w is not None]
        self._max_weight = max(valid_weights) if len(valid_weights) > 0 else 0
        self._names = names
        #self._names = map(str, range(len(self._weights))) if names is None else names
        #if len(weights) == 2:
        #    self.display_all = True

        # visualize each mesh
        self._trimeshes = []
        self._texts = []
        self._offsets = []
        offset = np.zeros(3)
        for i, (verts_i, tris_i, weights_i, name_i) in enumerate(zip(verts, tris, weights, names)):
            if i > 0:
                offset[offset_axis] += verts_i[:,offset_axis].ptp() * (1 + offset_spacing)
            if num_columns is not None and i % num_columns == 0:
                offset[offset_axis2] += verts_i[:, offset_axis2].ptp() * (1 + offset_spacing2)
                offset[offset_axis] = 0
            if center:
                vmin, vmax = 0, 1
            else:
                vmin, vmax = vmin, vmax
            # draw mesh
            tm = mlab.triangular_mesh(
                verts_i[:,0], verts_i[:,1], verts_i[:,2], tris_i, 
                scalars=weights_i[...,0] if weights_i is not None and not weights_i.dtype == np.uint8 else None, 
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

            if show_labels:
                txt = mlab.text(0, 0, str(name_i), z=0, color=(0, 0, 0), width=0.12)
                txt.actor.text_scale_mode = 'none'
                txt.property.set(font_size=11, justification='centered')
                self._texts.append(txt)


            #tm.actor.actor.property.edit_traits()
            # for the next mesh, add the extent of the current mesh (plus spacing) to the offset
            self._offsets.append(offset.copy())
            tm.actor.property.set(**actor_options)
            #if handle_points is not None:
            #    h = verts2.ptp()
            #    mlab.points3d(handle_points[:,0], handle_points[:,1], handle_points[:,2], scale_factor=h / 20)

        #actor_prop.edit_traits()
        self.selected_mesh = self._names[0]
        self._label_offset_axis = label_offset_axis
        self._label_offset = label_offset
        if self._label_offset_axis is None:
            self._label_offset_axis = (np.abs(self._offsets[-1]).argmax() + 1) % 3
        self._update_view()
        self._reposition_meshes()

    @on_trait_change('weight_index')
    def _update_view(self):
        for tms, w in zip(self._trimeshes, self._weights):
            tm = tms[0]
            if w is None:
                continue
            try:
                wi = w[...,self.weight_index]
                if wi.dtype == np.uint8:
                    tm.actor.mapper.input.point_data.scalars = wi
                    self.scene.render()
                else:
                    tm.mlab_source.set(scalars=wi)
            except IndexError:
                logging.warn("coult not reference index %d" % self.weight_index)

    @on_trait_change('display_all')
    def _reposition_meshes(self):
        self.scene.disable_render = True
        if self.display_all:
            for tms, offset in zip(self._trimeshes, self._offsets):
                for tm in tms:
                    tm.actor.actor.position = offset
            for txt, v, offset in zip(self._texts, self._verts, self._offsets):
                ax = np.zeros(3)
                ax[self._label_offset_axis] = self._label_offset
                txt.x_position, txt.y_position, txt.z_position = (v.mean(axis=0) + v.ptp(axis=0) * ax) + offset
        else:
            for tms in self._trimeshes:
                for tm in tms:
                    tm.actor.actor.position = (0, 0, 0)
        self.scene.disable_render = False

    @on_trait_change('display_all, selected_mesh, display_labels')
    def _update_mesh_visibilities(self):
        self.scene.disable_render = True
        for name, tms in zip(self._names, self._trimeshes):
            for tm in tms:
                tm.visible = self.display_all or name == self.selected_mesh
        for txt in self._texts:
            txt.visible = self.display_all and self.display_labels
        self.scene.disable_render = False

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
                 height=800, width=800, show_label=False),
        ),
        HGroup(
            Item('weight_index', label='idx'),
            Item('display_all', label='all'),
            Item('display_labels', label='labels', 
                visible_when='len(object._texts) > 0 and object.display_all'),
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

