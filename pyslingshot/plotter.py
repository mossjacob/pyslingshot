from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pcurvepy2 import PrincipalCurve

if TYPE_CHECKING:
    from .slingshot import Slingshot


def generate_colormap(number_of_distinct_colors: int = 80) -> list[tuple[float, float, float]]:
    """Generate visually distinct colors for cluster visualization.

    Uses seaborn's husl palette which provides maximally distinct colors
    in a perceptually uniform color space.
    """
    if number_of_distinct_colors == 0:
        number_of_distinct_colors = 80
    return sns.color_palette("husl", n_colors=number_of_distinct_colors)


class SlingshotPlotter:
    def __init__(self, sling: Slingshot) -> None:
        self.sling = sling

    def clusters(
        self,
        ax: Axes,
        labels: np.ndarray | None = None,
        s: int = 8,
        alpha: float = 1.0,
        color_mode: str = "clusters",
    ) -> None:
        from typing import Any as ColorType

        fig = plt.gcf()
        sling = self.sling
        if labels is None:
            labels = np.arange(sling.num_clusters)

        # Plot clusters and start cluster
        ax.scatter(sling.cluster_centres[sling.start_node][0], sling.cluster_centres[sling.start_node][1], c="red")

        colors: ColorType
        if color_mode == "clusters":
            palette = generate_colormap(sling.num_clusters)

            handles = [Patch(color=palette[k], label=str(labels[k])) for k in range(sling.num_clusters)]
            ax.legend(handles=handles)
            colors = [palette[i] for i in sling.cluster_label_indices]
        elif color_mode == "pseudotime":
            assert self.sling.curves is not None
            assert sling.lineages is not None
            colors = np.zeros_like(self.sling.curves[0].pseudotimes_interp)
            for l_idx, lineage in enumerate(sling.lineages):
                curve = self.sling.curves[l_idx]
                cell_mask = np.logical_or.reduce(np.array([sling.cluster_label_indices == k for k in lineage]))
                colors[cell_mask] = curve.pseudotimes_interp[cell_mask]
        elif type(color_mode) is np.ndarray:
            colors = color_mode
        else:
            colors = "black"

        main_scatter = ax.scatter(sling.data[:, 0], sling.data[:, 1], c=colors, s=s, alpha=alpha)

        if color_mode == "pseudotime":
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(main_scatter, cax=cax, orientation="vertical")

    def curves(self, ax: Axes, curves: list[PrincipalCurve]) -> None:
        for l_idx, curve in enumerate(curves):
            s_interp, p_interp, order = curve.unpack_params()
            ax.plot(p_interp[order, 0], p_interp[order, 1], label=f"Lineage {l_idx}", alpha=1)
            ax.legend()

    def network(self, cluster_to_label: dict[int, str], figsize: tuple[int, int] = (8, 10)) -> None:
        import networkx as nx
        from networkx.drawing.nx_agraph import graphviz_layout

        assert self.sling.lineages is not None

        plt.figure(figsize=figsize)
        G = nx.DiGraph(scale=0.02)
        lineages = self.sling.lineages
        root = cluster_to_label[lineages[0].clusters[0]]

        for lineage in lineages:
            parent = root
            for cluster_id in lineage:
                node = cluster_to_label[cluster_id]
                G.add_node(node)
                G.add_edge(parent, node)
                parent = node

        plt.title("Lineages")
        pos = graphviz_layout(G, prog="dot")
        label_options = {"ec": "k", "fc": "b", "alpha": 0.9, "boxstyle": "round,pad=0.2"}

        nx.draw(G, pos, arrows=True, node_size=[len(v) * 100 for v in G.nodes()])
        nx.draw_networkx_labels(G, pos, font_size=14, font_color="w", bbox=label_options)
