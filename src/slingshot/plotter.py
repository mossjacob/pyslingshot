import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable


class SlingshotPlotter:
    def __init__(self, sling):
        self.sling = sling

    def clusters(self, ax, labels=None, s=8, alpha=1., color_mode='clusters'):
        fig = plt.gcf()
        sling = self.sling
        if labels is None:
            labels = np.arange(sling.num_clusters)

        # Plot clusters and start cluster
        ax.scatter(
            sling.cluster_centres[sling.start_node][0],
            sling.cluster_centres[sling.start_node][1], c='red')

        if color_mode == 'clusters':
            colors = np.array(sns.color_palette())[sling.cluster_labels]
            handles = [
                Patch(color=colors[k], label=labels[k]) for k in range(sling.num_clusters)
            ]
            ax.legend(handles=handles)

        elif color_mode == 'pseudotime':
            colors = np.zeros_like(self.sling.curves[0].pseudotimes_interp)
            for l_idx, lineage in enumerate(sling.lineages):
                curve = self.sling.curves[l_idx]
                cell_mask = np.logical_or.reduce(
                    np.array([sling.cluster_labels == k for k in lineage]))
                colors[cell_mask] = curve.pseudotimes_interp[cell_mask]

        else:
            colors = 'black'

        main_scatter = ax.scatter(sling.data[:, 0], sling.data[:, 1],
                   c=colors,
                   s=s,
                   alpha=alpha)

        if color_mode == 'pseudotime':
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(main_scatter, cax=cax, orientation='vertical')

    def curves(self, ax, curves):
        for l_idx, curve in enumerate(curves):
            s_interp, p_interp, order = curve.unpack_params()
            ax.plot(
                p_interp[order, 0],
                p_interp[order, 1],
                label=f'Lineage {l_idx}',
                alpha=1)
            ax.legend()
