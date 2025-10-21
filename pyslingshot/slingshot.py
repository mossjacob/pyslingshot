from __future__ import annotations

from collections import deque

import numpy as np
from anndata import AnnData
from matplotlib.axes import Axes
from pcurvepy2 import PrincipalCurve
from scipy.interpolate import interp1d
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import KernelDensity
from tqdm.autonotebook import tqdm

from .lineage import Lineage
from .plotter import SlingshotPlotter
from .util import infer_cluster_label_indices, mahalanobis, scale_to_range


class Slingshot:
    def __init__(
        self,
        data: AnnData | np.ndarray,
        cluster_labels_onehot: np.ndarray | None = None,
        celltype_key: str | None = None,
        obsm_key: str = "X_umap",
        start_node: int = 0,
        end_nodes: list[int] | None = None,
        is_debugging: bool = False,
        approx_points: int | None = None,
    ) -> None:
        """
        Constructs a new `Slingshot` object.
        Args:
            data: either an AnnData object or a numpy array containing the dimensionality-reduced data of shape (num_cells, 2)
            cluster_labels_onehot: cluster assignments of shape (num_cells). Only required if `data` is not an AnnData object.
            celltype_key: key into AnnData.obs indicating cell type. Only required if `data` is an AnnData object.
            obsm_key: key into AnnData.obsm indicating the dimensionality-reduced data. Only required if `data` is an AnnData object.
            start_node: the starting node of the minimum spanning tree
            end_nodes: any terminal nodes
            is_debugging: whether to show debugging information and plots
            approx_points: number of approximate points to use when fitting the principal curves
        """
        if isinstance(data, AnnData):
            if celltype_key is None:
                raise ValueError("Must provide celltype key if data is an AnnData object")
            cluster_labels: np.ndarray = data.obs[celltype_key]

            self.cluster_label_indices = infer_cluster_label_indices(cluster_labels)
            num_cells = cluster_labels.shape[0]
            num_clusters = int(self.cluster_label_indices.max()) + 1
            _cluster_labels_onehot = np.zeros((num_cells, num_clusters))
            row_indices = np.arange(num_cells)
            col_indices: np.ndarray = self.cluster_label_indices.astype(int)
            _cluster_labels_onehot[row_indices, col_indices] = 1

            data = data.obsm[obsm_key]
        else:
            if cluster_labels_onehot is None:
                raise ValueError("Must provide cluster labels if data is not an AnnData object")
            cluster_labels = cluster_labels_onehot.argmax(axis=1)
            _cluster_labels_onehot = cluster_labels_onehot
            self.cluster_label_indices = infer_cluster_label_indices(cluster_labels)

        self.data = data
        self.approx_points = approx_points
        self.cluster_labels_onehot = _cluster_labels_onehot
        self.cluster_labels = cluster_labels
        self.num_clusters = self.cluster_label_indices.max() + 1
        self.start_node = start_node
        self.end_nodes = [] if end_nodes is None else end_nodes
        cluster_centres = [data[self.cluster_label_indices == k].mean(axis=0) for k in range(self.num_clusters)]
        self.cluster_centres = np.stack(cluster_centres)
        self.lineages: list[Lineage] | None = None
        self.cluster_lineages: dict[int, list[int]] | None = None
        self.curves: list[PrincipalCurve] | None = None
        self.cell_weights: np.ndarray | None = None
        self.distances: list[np.ndarray] | None = None
        self.branch_clusters: deque[int] | None = None
        self._tree: dict[int, list[int]] | None = None

        # Plotting and printing
        self.is_debugging = is_debugging
        self.debug_axes: np.ndarray | None = None
        self.debug_plot_mst: bool = False
        self.debug_plot_lineages: bool = False
        self.debug_plot_avg: bool = False
        self._set_debug_axes(None)
        self.plotter = SlingshotPlotter(self)

        # Construct smoothing kernel for the shrinking step
        self.kernel_x = np.linspace(-3, 3, 512)
        kde = KernelDensity(bandwidth=1.0, kernel="gaussian")
        kde.fit(np.zeros((self.kernel_x.shape[0], 1)))
        self.kernel_y = np.exp(kde.score_samples(self.kernel_x.reshape(-1, 1)))

    @property
    def tree(self) -> dict[int, list[int]]:
        if self._tree is None:
            self._tree = self.construct_mst(self.start_node)
        return self._tree

    def load_params(self, filepath: str) -> None:
        if self.curves is None:
            self.get_lineages()
        params = np.load(filepath, allow_pickle=True).item()
        self.curves = params["curves"]  # list of principle curves len = #lineages
        self.cell_weights = params["cell_weights"]  # weights indicating cluster assignments
        self.distances = params["distances"]

    def save_params(self, filepath: str) -> None:
        params = {"curves": self.curves, "cell_weights": self.cell_weights, "distances": self.distances}
        np.save(filepath, params, allow_pickle=True)

    def _set_debug_axes(self, axes: np.ndarray | None) -> None:
        self.debug_axes = axes
        self.debug_plot_mst = axes is not None
        self.debug_plot_lineages = axes is not None
        self.debug_plot_avg = axes is not None

    def _get_debug_ax(self, row: int, col: int) -> Axes:
        """Helper to extract an Axes object from debug_axes with proper typing.

        debug_axes is a 2x2 numpy array of matplotlib Axes objects.
        """
        if self.debug_axes is None:
            raise RuntimeError("debug_axes is not set")
        return self.debug_axes[row, col]  # type: ignore[return-value]

    def construct_mst(self, start_node: int) -> dict[int, list[int]]:
        """
        Parameters
           start_node: the starting node of the minimum spanning tree
        Returns:
             children: a dictionary mapping clusters to the children of each cluster
        """
        # Calculate empirical covariance of clusters
        emp_covs = np.stack([np.cov(self.data[self.cluster_label_indices == i].T) for i in range(self.num_clusters)])
        dists = np.zeros((self.num_clusters, self.num_clusters))
        for i in range(self.num_clusters):
            for j in range(i, self.num_clusters):
                dist = mahalanobis(self.cluster_centres[i], self.cluster_centres[j], emp_covs[i], emp_covs[j])
                dists[i, j] = dist
                dists[j, i] = dist

        # Find minimum spanning tree excluding end nodes
        mst_dists = np.delete(np.delete(dists, self.end_nodes, axis=0), self.end_nodes, axis=1)  # Delete end nodes
        tree = minimum_spanning_tree(mst_dists)
        # On the left: indices with ends removed; on the right: index into an array where the ends are skipped
        index_mapping = np.arange(self.num_clusters - len(self.end_nodes), dtype=int)
        for i, end_node in enumerate(self.end_nodes):
            index_mapping[end_node - i :] += 1

        connections: dict[int, list[int]] = {k: [] for k in range(self.num_clusters)}
        cx = tree.tocoo()
        for i, j in zip(cx.row, cx.col):
            i = index_mapping[i]
            j = index_mapping[j]
            connections[i].append(j)
            connections[j].append(i)

        for end in self.end_nodes:
            i = int(np.argmin(np.delete(dists[end], self.end_nodes)))
            connections[i].append(end)
            connections[end].append(i)

        # for i,j,v in zip(cx.row, cx.col, cx.data):
        visited = [False for _ in range(self.num_clusters)]
        queue: list[int] = []
        queue.append(start_node)
        children: dict[int, list[int]] = {k: [] for k in range(self.num_clusters)}
        while len(queue) > 0:  # BFS to construct children dict
            current_node = queue.pop()
            visited[current_node] = True
            for child in connections[current_node]:
                if not visited[child]:
                    children[current_node].append(child)
                    queue.append(child)

        # Plot clusters and MST
        if self.debug_plot_mst and self.debug_axes is not None:
            ax = self._get_debug_ax(0, 0)
            self.plotter.clusters(ax, alpha=0.5)
            for root, kids in children.items():
                for child_node in kids:
                    x_coords = [self.cluster_centres[root][0], self.cluster_centres[child_node][0]]
                    y_coords = [self.cluster_centres[root][1], self.cluster_centres[child_node][1]]
                    ax.plot(x_coords, y_coords, c="black")
            self.debug_plot_mst = False

        self._tree = children
        return children

    def fit(self, num_epochs: int = 10, debug_axes: np.ndarray | None = None) -> None:
        # Validate debug_axes shape if provided
        if debug_axes is not None:
            if debug_axes.shape != (2, 2):
                raise ValueError(f"debug_axes must be a 2x2 grid of Axes, got shape {debug_axes.shape}")

        self._set_debug_axes(debug_axes)

        # Initialize curves and cell weights if needed
        if self.curves is None:
            self.get_lineages()
            self.construct_initial_curves()

            # After get_lineages() and construct_initial_curves(), these are guaranteed to be set
            if self.lineages is None or self.cluster_labels_onehot is None:
                raise RuntimeError("Failed to initialize lineages")

            cell_weights_list = [
                self.cluster_labels_onehot[:, [int(c) for c in self.lineages[lineage_idx].clusters]].sum(axis=1)
                for lineage_idx in range(len(self.lineages))
            ]
            self.cell_weights = np.stack(cell_weights_list, axis=1)

        # Validate required attributes are set
        if self.lineages is None or self.curves is None or self.cell_weights is None:
            raise RuntimeError("Slingshot must be initialized before fitting")

        for epoch in tqdm(range(num_epochs)):
            # Calculate cell weights
            # cell weight is a matrix #cells x #lineages indicating cell-lineage assignment
            self.calculate_cell_weights()

            # Fit principal curve for all lineages using existing curves
            self.fit_lineage_curves()

            # Ensure starts at 0
            for l_idx in range(len(self.lineages)):
                curve = self.curves[l_idx]
                min_time = float(np.min(curve.pseudotimes_interp[self.cell_weights[:, l_idx] > 0]))
                curve.pseudotimes_interp -= min_time

            # Determine average curves
            shrinkage_percentages, cluster_children, cluster_avg_curves = self.avg_curves()

            # Shrink towards average curves in areas of cells common to all branch lineages
            self.shrink_curves(cluster_children, shrinkage_percentages, cluster_avg_curves)

            self.debug_plot_lineages = False
            self.debug_plot_avg = False

            if self.debug_axes is not None and epoch == num_epochs - 1:  # plot curves
                ax = self._get_debug_ax(1, 1)
                self.plotter.clusters(ax, s=2, alpha=0.5)
                self.plotter.curves(ax, self.curves)

    def construct_initial_curves(self) -> None:
        """Constructs lineage principal curves using piecewise linear initialisation"""
        if self.lineages is None:
            raise ValueError("Lineages must be initialised with `get_lineages()` before constructing initial curves.")
        piecewise_linear = []
        distances = []

        for lineage in self.lineages:
            # Calculate piecewise linear path
            p = np.stack(self.cluster_centres[lineage.clusters])

            cell_mask = np.logical_or.reduce(np.array([self.cluster_label_indices == k for k in lineage]))
            cells_involved = self.data[cell_mask]

            curve = PrincipalCurve(k=3)
            curve.project_to_curve(cells_involved, points=p)
            d_sq, dist = curve.project_to_curve(self.data, points=curve.points_interp[curve.order])
            distances.append(d_sq)

            piecewise_linear.append(curve)

        self.curves = piecewise_linear
        self.distances = distances

    def get_lineages(self) -> None:
        tree = self.construct_mst(self.start_node)

        # Determine lineages by parsing the MST
        branch_clusters: deque[int] = deque()

        def recurse_branches(path, v):
            num_children = len(tree[v])
            if num_children == 0:  # at leaf, add a None token
                return path + [v, None]
            elif num_children == 1:
                return recurse_branches(path + [v], tree[v][0])
            else:  # at branch
                branch_clusters.append(v)
                return [recurse_branches(path + [v], tree[v][i]) for i in range(num_children)]

        def flatten(li):
            if li[-1] is None:  # special None token indicates a leaf
                yield Lineage(li[:-1])
            else:  # otherwise yield from children
                for item in li:
                    yield from flatten(item)

        lineages = recurse_branches([], self.start_node)
        lineages = list(flatten(lineages))
        self.lineages = lineages
        self.branch_clusters = branch_clusters

        self.cluster_lineages = {k: [] for k in range(self.num_clusters)}
        for l_idx, lineage in enumerate(lineages):
            for k in lineage:
                self.cluster_lineages[k].append(l_idx)

        if self.is_debugging:
            print("Lineages:", lineages)

    def fit_lineage_curves(self) -> None:
        """Updates curve using a cubic spline and projection of data"""
        assert self.lineages is not None
        assert self.curves is not None
        assert self.cell_weights is not None
        distances: list[np.ndarray] = []

        # Calculate principal curves
        for l_idx, lineage in enumerate(self.lineages):
            curve: PrincipalCurve = self.curves[l_idx]

            # Fit principal curve through data
            # Weights are important as they effectively silence points
            # that are not associated with the lineage.
            curve.fit(self.data, max_iter=1, weights=self.cell_weights[:, l_idx], approx_points=self.approx_points)

            if self.debug_plot_lineages and self.debug_axes is not None:
                ax = self._get_debug_ax(0, 1)
                cell_mask = np.logical_or.reduce(np.array([self.cluster_label_indices == k for k in lineage]))
                cells_involved = self.data[cell_mask]
                ax.scatter(cells_involved[:, 0], cells_involved[:, 1], s=2, alpha=0.5)
                alphas = curve.pseudotimes_interp
                alphas = (alphas - alphas.min()) / (alphas.max() - alphas.min())
                for i in np.random.permutation(self.data.shape[0])[:50]:
                    path_from = (self.data[i][0], curve.points_interp[i][0])
                    path_to = (self.data[i][1], curve.points_interp[i][1])
                    ax.plot(path_from, path_to, c="black", alpha=alphas[i])
                ax.plot(curve.points_interp[curve.order, 0], curve.points_interp[curve.order, 1], label=str(lineage))

            d_sq, dist = curve.project_to_curve(self.data, curve.points_interp[curve.order])
            distances.append(d_sq)
        self.distances = distances
        if self.debug_plot_lineages and self.debug_axes is not None:
            ax = self._get_debug_ax(0, 1)
            ax.legend()

    def calculate_cell_weights(self) -> None:
        """Calculate soft assignment weights for cells to lineages.

        This implements a probabilistic weighting scheme that assigns cells to multiple
        lineages based on their proximity to each lineage path. The algorithm:

        1. Initializes weights based on cluster membership (cells get weight 1 for lineages
           containing their cluster, 0 otherwise)
        2. Converts distances to percentile ranks to create a normalized distance metric
        3. Transforms percentile ranks into weights using 1 - rank^2
        4. Normalizes weights so the maximum weight per cell is 1
        5. Applies reassignment rules:
           - Cells very close to a lineage (rank < 0.5) get full weight (1.0)
           - Cells far from a lineage (rank > 0.9) with low weight (<0.1) get zero weight

        The result is a soft assignment where each cell can belong to multiple lineages
        with varying degrees of membership, enabling smooth transitions at branch points.

        This is a translation from the R version, with the following variable name clarifications:
          - cell_weights → initial_weights (Step 1)
          - d_sq → lineage_distances
          - d_ord → sorted_distance_indices
          - w_prob → weight_probabilities
          - w_rnk_d → cumulative_weight_ranks
          - z → distance_percentiles
          - z_prime → weight_scores
          - w0 → initial_weights_backup
          - z0 → percentiles_to_reassign
          - ridx → cells_to_reassign_mask
          - reassign → reassign_enabled
        """
        assert self.lineages is not None
        assert self.distances is not None

        # Step 1: Initialize weights based on cluster membership
        # For each lineage, sum the one-hot cluster indicators for clusters in that lineage
        # This gives binary weights: 1 if cell's cluster is in the lineage, 0 otherwise
        initial_weights_list = [
            self.cluster_labels_onehot[:, [int(c) for c in self.lineages[lineage_idx].clusters]].sum(axis=1)
            for lineage_idx in range(len(self.lineages))
        ]
        initial_weights = np.stack(initial_weights_list, axis=1)  # shape: (num_cells, num_lineages)

        # Step 2: Prepare distance matrix and compute distance-based percentile ranks
        # distances is a list of arrays (one per lineage), stack into single matrix
        lineage_distances = np.stack(self.distances, axis=1)  # shape: (num_cells, num_lineages)

        # Get indices that would sort all distances (flattened) in ascending order
        sorted_distance_indices = np.argsort(lineage_distances, axis=None)

        # Convert binary weights to probability distribution (normalize per cell)
        weight_probabilities = initial_weights / initial_weights.sum(axis=1, keepdims=True)

        # Compute cumulative weight distribution in order of increasing distance
        # This creates a percentile rank for each distance value, weighted by initial cluster assignment
        cumulative_weight_ranks = np.cumsum(weight_probabilities.reshape(-1)[sorted_distance_indices])
        cumulative_weight_ranks = cumulative_weight_ranks / weight_probabilities.sum()

        # Step 3: Map distances to their percentile ranks
        # Replace each distance value with its weighted percentile rank (0 to 1)
        distance_percentiles = lineage_distances.copy()
        original_shape = distance_percentiles.shape
        distance_percentiles = distance_percentiles.reshape(-1)
        distance_percentiles[sorted_distance_indices] = cumulative_weight_ranks
        distance_percentiles = distance_percentiles.reshape(original_shape)

        # Step 4: Transform percentile ranks into weight scores
        # Use 1 - rank^2 transformation (quadratic decay from 1 to 0)
        weight_scores = 1 - distance_percentiles**2

        # Set weight to NaN for lineages that don't contain the cell's cluster
        weight_scores[initial_weights == 0] = np.nan

        # Backup initial weights for later validation
        initial_weights_backup = initial_weights.copy()

        # Step 5: Normalize weights so maximum weight per cell is 1
        cell_weights = weight_scores / np.nanmax(weight_scores, axis=1, keepdims=True)

        # Handle edge case: cells equidistant from all lineages (0/0 = NaN)
        # Give such cells weight 1 for all lineages
        np.nan_to_num(cell_weights, nan=1, copy=False)

        # Clamp weights to valid range [0, 1]
        cell_weights[cell_weights > 1] = 1
        cell_weights[cell_weights < 0] = 0

        # Ensure cells get 0 weight for lineages they weren't initially assigned to
        cell_weights[initial_weights_backup == 0] = 0

        # Step 6: Apply reassignment rules for cells near branch points
        reassign_enabled = True
        if reassign_enabled:
            # Rule 1: Cells very close to a lineage (low percentile rank) get full weight
            # If distance percentile < 0.5, set weight to 1 (cell is definitively on this lineage)
            cell_weights[distance_percentiles < 0.5] = 1

            # Rule 2: Remove weak assignments for cells far from all lineages
            # Identify cells where: max distance percentile > 0.9 AND min weight < 0.1
            # These are cells that are far from all lineages but have a spurious weak assignment
            cells_to_reassign_mask = (distance_percentiles.max(axis=1) > 0.9) & (cell_weights.min(axis=1) < 0.1)

            # For these cells, zero out any lineage weight where distance > 0.9 and weight < 0.1
            weights_to_reassign = cell_weights[cells_to_reassign_mask]
            percentiles_to_reassign = distance_percentiles[cells_to_reassign_mask]
            weights_to_reassign[(percentiles_to_reassign > 0.9) & (weights_to_reassign < 0.1)] = 0
            cell_weights[cells_to_reassign_mask] = weights_to_reassign

        self.cell_weights = cell_weights

    def avg_curves(
        self,
    ) -> tuple[list[dict[PrincipalCurve, np.ndarray]], dict[int, set[PrincipalCurve]], dict[int, PrincipalCurve]]:
        """
        Starting at leaves, calculate average curves for each branch

        :return: shrinkage_percentages, cluster_children, cluster_avg_curves
        """
        assert self.curves is not None
        assert self.branch_clusters is not None
        assert self.cluster_lineages is not None
        assert self.cell_weights is not None

        shrinkage_percentages: list[dict[PrincipalCurve, np.ndarray]] = []
        cluster_children: dict[int, set[PrincipalCurve]] = {}
        lineage_avg_curves: dict[int, PrincipalCurve] = {}
        cluster_avg_curves: dict[int, PrincipalCurve] = {}
        branch_clusters = self.branch_clusters.copy()
        if self.is_debugging:
            print("Reversing from leaf to root")
        if self.debug_plot_avg and self.debug_axes is not None:
            ax = self._get_debug_ax(1, 0)
            self.plotter.clusters(ax, s=4, alpha=0.4)

        while len(branch_clusters) > 0:
            # Starting at leaves, find lineages involved in branch
            k = branch_clusters.pop()
            branch_lineages = self.cluster_lineages[k]
            cluster_children[k] = set()
            for l_idx in branch_lineages:  # loop all lineages through branch
                if l_idx in lineage_avg_curves:  # add avg curve
                    curve = lineage_avg_curves[l_idx]
                else:  # or add leaf curve
                    curve = self.curves[l_idx]
                cluster_children[k].add(curve)

            # Calculate the average curve for this branch
            branch_curves = list(cluster_children[k])
            if self.is_debugging:
                print(f"Averaging branch @{k} with lineages:", branch_lineages, branch_curves)

            avg_curve = self.avg_branch_curves(branch_curves)
            cluster_avg_curves[k] = avg_curve
            # avg.curve$w <- rowSums(vapply(pcurves, function(p){ p$w }, rep(0,nrow(X))))

            # Calculate shrinkage weights using areas where cells share lineages
            # note that this also captures cells in average curves, since the
            # lineages which are averaged are present in branch_lineages
            common = self.cell_weights[:, branch_lineages] > 0
            common_mask = common.mean(axis=1) == 1.0
            shrinkage_percent: dict[PrincipalCurve, np.ndarray] = {}
            for curve in branch_curves:
                shrinkage_percent[curve] = self.shrinkage_percent(curve, common_mask)
            shrinkage_percentages.append(shrinkage_percent)

            # Add avg_curve to lineage_avg_curve for cluster_children
            for lineage_idx in branch_lineages:
                lineage_avg_curves[lineage_idx] = avg_curve
            # # check for degenerate case (if one curve won't be
            # # shrunk, then the other curve shouldn't be,
            # # either)
            # new.avg.order <- avg.order
            # all.zero <- vapply(pct.shrink[[i]], function(pij){
            #     return(all(pij == 0))
            # }, TRUE)
            # if(any(all.zero)){
            #     if(allow.breaks){
            #         new.avg.order[[i]] <- NULL
            #         message('Curves for ', ns[1], ' and ',
            #             ns[2], ' appear to be going in opposite ',
            #             'directions. No longer forcing them to ',
            #             'share an initial point. To manually ',
            #             'override this, set allow.breaks = ',
            #             'FALSE.')
            #     }
            #     pct.shrink[[i]] <- lapply(pct.shrink[[i]],
            #         function(pij){
            #             pij[] <- 0
            #             return(pij)
            #         })
            # }
        if self.debug_plot_avg and self.debug_axes is not None:
            ax = self._get_debug_ax(1, 0)
            ax.legend()
        return shrinkage_percentages, cluster_children, cluster_avg_curves

    def shrink_curves(
        self,
        cluster_children: dict[int, set[PrincipalCurve]],
        shrinkage_percentages: list[dict[PrincipalCurve, np.ndarray]],
        cluster_avg_curves: dict[int, PrincipalCurve],
    ) -> None:
        """
        Starting at root, shrink curves for each branch

        Parameters:
            cluster_children:
            shrinkage_percentages:
            cluster_avg_curves:
        :return:
        """
        assert self.branch_clusters is not None
        branch_clusters = self.branch_clusters.copy()
        while len(branch_clusters) > 0:
            # Starting at root, find lineages involves in branch
            k = branch_clusters.popleft()
            shrinkage_percent = shrinkage_percentages.pop()
            branch_curves = list(cluster_children[k])
            cluster_avg_curve = cluster_avg_curves[k]
            if self.is_debugging:
                print(f"Shrinking branch @{k} with curves:", branch_curves)

            # Specify the avg curve for this branch
            self.shrink_branch_curves(branch_curves, cluster_avg_curve, shrinkage_percent)

    def shrink_branch_curves(self, branch_curves, avg_curve, shrinkage_percent):
        """
        Shrinks curves through a branch to the average curve.

        :param branch_curves: list of `PrincipalCurve`s associated with the branch.
        :param avg_curve: `PrincipalCurve` for average curve.
        :param shrinkage_percent: percentage shrinkage, in same order as curve.pseudotimes
        """
        num_dims_reduced = branch_curves[0].points_interp.shape[1]

        # Go through "child" lineages, shrinking the curves toward the above average
        for curve in branch_curves:  # curve might be an average curve or a leaf curve
            pct = shrinkage_percent[curve]

            s_interp, p_interp, order = curve.unpack_params()
            avg_s_interp, avg_p_interp, avg_order = avg_curve.unpack_params()
            shrunk_curve = np.zeros_like(p_interp)
            for j in range(num_dims_reduced):
                orig = p_interp[order, j]
                avg = np.interp(  # interp1d(
                    s_interp[order],
                    avg_s_interp[avg_order],  # x
                    avg_p_interp[avg_order, j],
                )  # ,  # y
                # assume_sorted=True,
                # bounds_error=False,
                # fill_value='extrapolate',
                # extrapolate_extrema=True)
                # avg = lin_interpolator#(s_interp[order])
                shrunk_curve[:, j] = avg * pct + orig * (1 - pct)
            # w <- pcurve$w
            # pcurve = project_to_curve(X, as.matrix(s[pcurve$ord, ,drop = FALSE]), stretch = stretch)
            # pcurve$w <- w
            # self.debug_axes[1, 1].plot(
            #     shrunk_curve[:, 0],
            #     shrunk_curve[:, 1],
            #     label='shrunk', alpha=0.2, c='black')
            curve.project_to_curve(self.data, points=shrunk_curve)
            #     for(jj in seq_along(ns)){
            #         n <- ns[jj]
            #         if(grepl('Lineage',n)){
            #             l.ind <- as.numeric(gsub('Lineage','',n))
            #             pcurves[[l.ind]] <- shrunk[[jj]]
            #         }
            #         if(grepl('average',n)){
            #             a.ind <- as.numeric(gsub('average','',n))
            #             avg.lines[[a.ind]] <- shrunk[[jj]]
            #         }
            #     }
            # }
            # avg.order <- new.avg.order

    def shrinkage_percent(self, curve, common_ind):
        """Determines how much to shrink a curve"""
        # pst <- crv$lambda
        # pts2wt <- pst
        s_interp, order = curve.pseudotimes_interp, curve.order
        # Cosine kernel quartiles:
        x = self.kernel_x
        y = self.kernel_y
        y = (y.sum() - np.cumsum(y)) / sum(y)
        q1 = np.percentile(s_interp[common_ind], 25)
        q3 = np.percentile(s_interp[common_ind], 75)
        a = q1 - 1.5 * (q3 - q1)
        b = q3 + 1.5 * (q3 - q1)
        x = scale_to_range(x, a=a, b=b)
        if q1 == q3:
            pct_l = np.zeros(s_interp.shape[0])
        else:
            pct_l = np.interp(s_interp[order], x, y)

        return pct_l

    def avg_branch_curves(self, branch_curves):
        """branch_lineages is a list of lineages passing through branch"""
        # s_interps, p_interps, orders
        num_cells = branch_curves[0].points_interp.shape[0]
        num_dims_reduced = branch_curves[0].points_interp.shape[1]

        # 1. Interpolate all the lineages over the shared time domain
        branch_s_interps = np.stack([c.pseudotimes_interp for c in branch_curves], axis=1)
        # Take minimum of maximum pseudotimes for each lineage
        max_shared_pseudotime = branch_s_interps.max(axis=0).min()
        combined_pseudotime = np.linspace(0, max_shared_pseudotime, num_cells)
        curves_dense_list: list[np.ndarray] = []
        for curve in branch_curves:
            lineage_curve = np.zeros((combined_pseudotime.shape[0], num_dims_reduced))
            order = curve.order
            # Linearly interpolate each dimension as a function of pseudotime
            for j in range(num_dims_reduced):
                lin_interpolator = interp1d(
                    curve.pseudotimes_interp[order],  # x
                    curve.points_interp[order, j],  # y
                    assume_sorted=True,
                )
                lineage_curve[:, j] = lin_interpolator(combined_pseudotime)
            curves_dense_list.append(lineage_curve)

        curves_dense = np.stack(curves_dense_list, axis=1)  # (n, L_b, J)

        # 2. Average over these curves and project the data onto the result
        avg = curves_dense.mean(axis=1)  # avg is already "sorted"
        avg_curve = PrincipalCurve()
        avg_curve.project_to_curve(self.data, points=avg)
        # avg_curve.pseudotimes_interp -= avg_curve.pseudotimes_interp.min()
        if self.debug_plot_avg and self.debug_axes is not None:
            ax = self._get_debug_ax(1, 0)
            ax.plot(avg[:, 0], avg[:, 1], c="blue", linestyle="--", label="average", alpha=0.7)
            _, p_interp, order = avg_curve.unpack_params()
            ax.plot(p_interp[order, 0], p_interp[order, 1], c="red", label="data projected", alpha=0.7)

        # avg.curve$w <- rowSums(vapply(pcurves, function(p){ p$w }, rep(0,nrow(X))))
        return avg_curve

    @property
    def unified_pseudotime(self) -> np.ndarray:
        assert self.curves is not None
        assert self.lineages is not None
        pseudotime = np.zeros_like(self.curves[0].pseudotimes_interp)
        for l_idx, lineage in enumerate(self.lineages):
            curve = self.curves[l_idx]
            cell_mask = np.logical_or.reduce(np.array([self.cluster_label_indices == k for k in lineage]))
            pseudotime[cell_mask] = curve.pseudotimes_interp[cell_mask]
        return pseudotime

    def list_lineages(self, cluster_to_label: dict[int, str]) -> None:
        if self.lineages is None:
            raise ValueError("No lineages available")

        for lineage in self.lineages:
            print(", ".join([cluster_to_label[cluster_id] for cluster_id in lineage]))
