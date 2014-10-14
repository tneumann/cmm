import numpy as np
import matplotlib
import pylab as pl


def _htmlhex_to_rgb(col):
    return map(lambda h: float(int(h, 16)) / 255., (col[0:2], col[2:4], col[4:6]))


def nice_mpl_color_map():
    color1 = _htmlhex_to_rgb('0E8D84')
    color2 = _htmlhex_to_rgb('E67117')
    color_zero = (1., 1., 0.99)

    cdict = {'red': ((0.0, color1[0], color1[0]),
                     (0.5, color_zero[0], color_zero[0]),
                     (1.0, color2[0], color2[0])),
             'green': ((0.0, color1[1], color1[1]),
                       (0.5, color_zero[1], color_zero[1]),
                       (1.0, color2[1], color2[1])),
             'blue': ((0.0, color1[2], color1[2]),
                      (0.5, color_zero[2], color_zero[2]),
                      (1.0, color2[2], color2[2]))}
    return matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)


def plot_functional_map(C, newfig=True):
    vmax = max(np.abs(C.max()), np.abs(C.min()))
    vmin = -vmax
    C = ((C - vmin) / (vmax - vmin)) * 2 - 1
    if newfig:
        pl.figure(figsize=(5,5))
    else:
        pl.clf()
    ax = pl.gca()
    pl.pcolor(C[::-1], edgecolor=(0.9, 0.9, 0.9, 1), lw=0.5,
              vmin=-1, vmax=1, cmap=nice_mpl_color_map())
    # colorbar
    tick_locs   = [-1., 0.0, 1.0]
    tick_labels = ['min', 0, 'max']
    bar = pl.colorbar()
    bar.locator = matplotlib.ticker.FixedLocator(tick_locs)
    bar.formatter = matplotlib.ticker.FixedFormatter(tick_labels)
    bar.update_ticks()
    ax.set_aspect(1)
    pl.xticks([])
    pl.yticks([])
    if newfig:
        pl.show()
