import numpy as np
import seaborn as sns

from matplotlib.patches import Patch

# from .slingshot import Slingshot


class SlingshotPlotter:
    def __init__(self, sling):
        self.sling = sling

    def clusters(self, ax, labels=None, s=8, alpha=1.):
        sling = self.sling
        if labels is None:
            labels = np.arange(sling.num_clusters)

        # Plot clusters and start cluster
        colors = np.array(sns.color_palette())
        ax.scatter(sling.data[:, 0], sling.data[:, 1],
                   c=colors[sling.cluster_labels],
                   s=s,
                   alpha=alpha)
        ax.scatter(
            sling.cluster_centres[sling.start_node][0],
            sling.cluster_centres[sling.start_node][1], c='red')
        handles = [
            Patch(color=colors[k], label=labels[k]) for k in range(sling.num_clusters)
        ]
        ax.legend(handles=handles)

    def curves(self, ax, curves):
        for l_idx, curve in enumerate(curves):
            s_interp, p_interp, order = curve.unpack_params()
            ax.plot(
                p_interp[order, 0],
                p_interp[order, 1],
                label=f'Lineage {l_idx}',
                alpha=1)
            ax.legend()