import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

import math
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv


def generate_colormap(number_of_distinct_colors: int = 80):
    if number_of_distinct_colors == 0:
        number_of_distinct_colors = 80

    number_of_shades = 7
    number_of_distinct_colors_with_multiply_of_shades = int(math.ceil(number_of_distinct_colors / number_of_shades) * number_of_shades)

    # Create an array with uniformly drawn floats taken from <0, 1) partition
    linearly_distributed_nums = np.arange(number_of_distinct_colors_with_multiply_of_shades) / number_of_distinct_colors_with_multiply_of_shades

    # We are going to reorganise monotonically growing numbers in such way that there will be single array with saw-like pattern
    #     but each saw tooth is slightly higher than the one before
    # First divide linearly_distributed_nums into number_of_shades sub-arrays containing linearly distributed numbers
    arr_by_shade_rows = linearly_distributed_nums.reshape(number_of_shades, number_of_distinct_colors_with_multiply_of_shades // number_of_shades)

    # Transpose the above matrix (columns become rows) - as a result each row contains saw tooth with values slightly higher than row above
    arr_by_shade_columns = arr_by_shade_rows.T

    # Keep number of saw teeth for later
    number_of_partitions = arr_by_shade_columns.shape[0]

    # Flatten the above matrix - join each row into single array
    nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)

    # HSV colour map is cyclic (https://matplotlib.org/tutorials/colors/colormaps.html#cyclic), we'll use this property
    initial_cm = hsv(nums_distributed_like_rising_saw)

    lower_partitions_half = number_of_partitions // 2
    upper_partitions_half = number_of_partitions - lower_partitions_half

    # Modify lower half in such way that colours towards beginning of partition are darker
    # First colours are affected more, colours closer to the middle are affected less
    lower_half = lower_partitions_half * number_of_shades
    for i in range(3):
        initial_cm[0:lower_half, i] *= np.arange(0.2, 1, 0.8/lower_half)

    # Modify second half in such way that colours towards end of partition are less intense and brighter
    # Colours closer to the middle are affected less, colours closer to the end are affected more
    for i in range(3):
        for j in range(upper_partitions_half):
            modifier = np.ones(number_of_shades) - initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i]
            modifier = j * modifier / upper_partitions_half
            initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i] += modifier

    return ListedColormap(initial_cm)

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
            colors = np.array(sns.color_palette('cubehelix', n_colors=sling.num_clusters))
            colors = generate_colormap(sling.num_clusters)

            handles = [
                Patch(color=colors.colors[k], label=labels[k]) for k in range(sling.num_clusters)
            ]
            ax.legend(handles=handles)
            colors = colors.colors[sling.cluster_labels]
        elif color_mode == 'pseudotime':
            colors = np.zeros_like(self.sling.curves[0].pseudotimes_interp)
            for l_idx, lineage in enumerate(sling.lineages):
                curve = self.sling.curves[l_idx]
                cell_mask = np.logical_or.reduce(
                    np.array([sling.cluster_labels == k for k in lineage]))
                colors[cell_mask] = curve.pseudotimes_interp[cell_mask]
        elif type(color_mode) is np.array:
            colors = color_mode
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

    def network(self, cluster_to_label, figsize=(8, 10)):
        import networkx as nx
        from networkx.drawing.nx_agraph import graphviz_layout
        plt.figure(figsize=figsize)
        G = nx.DiGraph(scale=0.02)
        lineages = self.sling.lineages
        root = cluster_to_label[lineages[0].clusters[0]]

        for lineage in lineages:
            parent = root
            for l in lineage:
                node = cluster_to_label[l]
                G.add_node(node)
                G.add_edge(parent, node)
                parent = node

        plt.title('Lineages')
        pos = graphviz_layout(G, prog='dot')
        label_options = dict(
            ec="k", fc='b', alpha=0.9,
            boxstyle='round,pad=0.2'
        )

        nx.draw(
            G, pos,
            arrows=True,
            node_size=[len(v) * 100 for v in G.nodes()]
        )
        nx.draw_networkx_labels(G, pos, font_size=14, font_color='w', bbox=label_options)
