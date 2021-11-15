import numpy as np

from pcurve import PrincipalCurve
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.interpolate import interp1d
from sklearn.neighbors import KernelDensity
from collections import deque
from tqdm import tqdm

from .util import scale_to_range, mahalanobis
from .lineage import Lineage
from .plotter import SlingshotPlotter


class Slingshot():
    def __init__(self, data, cluster_labels, start_node=0, end_nodes=None, debug_level=None):
        self.data = data
        self.cluster_labels_onehot = cluster_labels
        self.cluster_labels = self.cluster_labels_onehot.argmax(axis=1)
        self.num_clusters = self.cluster_labels.max() + 1
        self.start_node = start_node
        self.end_nodes = [] if end_nodes is None else end_nodes
        cluster_centres = [data[self.cluster_labels == k].mean(axis=0) for k in range(self.num_clusters)]
        self.cluster_centres = np.stack(cluster_centres)
        self.lineages = None      # list of Lineages
        self.cluster_lineages = None # lineages belonging to each cluster
        self.curves = None   # list of principle curves len = #lineages
        self.cell_weights = None  # weights indicating cluster assignments
        self.distances = None
        self.branch_clusters = None

        # Plotting and printing
        debug_level = 0 if debug_level is None else dict(verbose=1)[debug_level]
        self.debug_level = debug_level
        self._set_debug_axes(None)
        self.plotter = SlingshotPlotter(self)

        # Construct smoothing kernel for the shrinking step
        self.kernel_x = np.linspace(-3, 3, 512)
        kde = KernelDensity(bandwidth=1., kernel='gaussian')
        kde.fit(np.zeros((self.kernel_x.shape[0], 1)))
        self.kernel_y = np.exp(kde.score_samples(self.kernel_x.reshape(-1, 1)))

    def load_params(self, filepath):
        if self.curves is None:
            self.get_lineages()
        params = np.load(filepath, allow_pickle=True).item()
        self.curves = params['curves']   # list of principle curves len = #lineages
        self.cell_weights = params['cell_weights']  # weights indicating cluster assignments
        self.distances = params['distances']

    def save_params(self, filepath):
        params = dict(
            curves=self.curves,
            cell_weights=self.cell_weights,
            distances=self.distances
        )
        np.save(filepath, params)

    def _set_debug_axes(self, axes):
        self.debug_axes = axes
        self.debug_plot_mst = axes is not None
        self.debug_plot_lineages = axes is not None
        self.debug_plot_avg = axes is not None

    def construct_mst(self, start_node):
        """
        Parameters
           start_node: the starting node of the minimum spanning tree
        Returns:
             children: a dictionary mapping clusters to the children of each cluster
        """
        # Calculate empirical covariance of clusters
        emp_covs = np.stack([np.cov(self.data[self.cluster_labels == i].T) for i in range(self.num_clusters)])
        dists = np.zeros((self.num_clusters, self.num_clusters))
        for i in range(self.num_clusters):
            for j in range(i, self.num_clusters):
                dist = mahalanobis(
                    self.cluster_centres[i],
                    self.cluster_centres[j],
                    emp_covs[i],
                    emp_covs[j]
                )
                dists[i, j] = dist
                dists[j, i] = dist

        # Find minimum spanning tree excluding end nodes
        mst_dists = np.delete(np.delete(dists, self.end_nodes, axis=0), self.end_nodes, axis=1)  # Delete end nodes
        tree = minimum_spanning_tree(mst_dists)
        # On the left: indices with ends removed; on the right: index into an array where the ends are skipped
        index_mapping = np.array([c for c in range(self.num_clusters - len(self.end_nodes))])
        for i, end_node in enumerate(self.end_nodes):
            index_mapping[end_node - i:] += 1

        connections = {k: list() for k in range(self.num_clusters)}
        cx = tree.tocoo()
        for i, j, v in zip(cx.row, cx.col, cx.data):
            i = index_mapping[i]
            j = index_mapping[j]
            connections[i].append(j)
            connections[j].append(i)

        for end in self.end_nodes:
            i = np.argmin(np.delete(dists[end], self.end_nodes))
            connections[i].append(end)
            connections[end].append(i)

        # for i,j,v in zip(cx.row, cx.col, cx.data):
        visited = [False for _ in range(self.num_clusters)]
        queue = list()
        queue.append(start_node)
        children = {k: list() for k in range(self.num_clusters)}
        while len(queue) > 0: # BFS to construct children dict
            current_node = queue.pop()
            visited[current_node] = True
            for child in connections[current_node]:
                if not visited[child]:
                    children[current_node].append(child)
                    queue.append(child)

        # Plot clusters and MST
        if self.debug_plot_mst:
            self.plotter.clusters(self.debug_axes[0, 0], alpha=0.5)
            for root, kids in children.items():
                for child in kids:
                    start = [self.cluster_centres[root][0], self.cluster_centres[child][0]]
                    end = [self.cluster_centres[root][1], self.cluster_centres[child][1]]
                    self.debug_axes[0, 0].plot(start, end, c='black')
            self.debug_plot_mst = False
        return children

    def fit(self, num_epochs=10, debug_axes=None):
        self._set_debug_axes(debug_axes)
        if self.curves is None:  # Initial curves and pseudotimes:
            self.get_lineages()
            self.construct_initial_curves()
            self.cell_weights = [self.cluster_labels_onehot[:, self.lineages[l].clusters].sum(axis=1)
                                 for l in range(len(self.lineages))]
            self.cell_weights = np.stack(self.cell_weights, axis=1)

        for epoch in tqdm(range(num_epochs)):
            # Calculate cell weights
            # cell weight is a matrix #cells x #lineages indicating cell-lineage assignment
            self.calculate_cell_weights()

            # Fit principal curve for all lineages using existing curves
            self.fit_lineage_curves()

            # Ensure starts at 0
            for l_idx, lineage in enumerate(self.lineages):
                curve = self.curves[l_idx]
                min_time = np.min(curve.pseudotimes_interp[self.cell_weights[:, l_idx] > 0])
                curve.pseudotimes_interp -= min_time

            # Determine average curves
            shrinkage_percentages, cluster_children, cluster_avg_curves = \
                self.avg_curves()

            # Shrink towards average curves in areas of cells common to all branch lineages
            self.shrink_curves(cluster_children, shrinkage_percentages, cluster_avg_curves)

            self.debug_plot_lineages = False
            self.debug_plot_avg = False

            if epoch == num_epochs - 1:  # plot curves
                self.plotter.clusters(self.debug_axes[1, 1], s=2, alpha=0.5)
                self.plotter.curves(self.debug_axes[1, 1], self.curves)

    def construct_initial_curves(self):
        """Constructs lineage principal curves using piecewise linear initialisation"""
        piecewise_linear = list()
        distances = list()

        for l_idx, lineage in enumerate(self.lineages):
            # Calculate piecewise linear path
            p = np.stack(self.cluster_centres[lineage.clusters])
            s = np.zeros(p.shape[0])  # TODO

            cell_mask = np.logical_or.reduce(
                np.array([self.cluster_labels == k for k in lineage]))
            cells_involved = self.data[cell_mask]

            curve = PrincipalCurve(k=3)
            curve.project_to_curve(cells_involved, points=p)
            d_sq, dist = curve.project_to_curve(self.data, points=curve.points_interp[curve.order])
            distances.append(d_sq)

            # piecewise_linear.append(PrincipalCurve.from_params(s, p))
            piecewise_linear.append(curve)

        self.curves = piecewise_linear
        self.distances = distances

    def get_lineages(self):
        tree = self.construct_mst(self.start_node)

        # Determine lineages by parsing the MST
        branch_clusters = deque()
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
                for l in li:
                    yield from flatten(l)

        lineages = recurse_branches([], self.start_node)
        lineages = list(flatten(lineages))
        self.lineages = lineages
        self.branch_clusters = branch_clusters

        self.cluster_lineages = {k: list() for k in range(self.num_clusters)}
        for l_idx, lineage in enumerate(self.lineages):
            for k in lineage:
                self.cluster_lineages[k].append(l_idx)

        if self.debug_level > 0:
            print('Lineages:', lineages)

    def fit_lineage_curves(self):
        """Updates curve using a cubic spline and projection of data"""
        assert self.lineages is not None
        assert self.curves is not None
        distances = list()

        # Calculate principal curves
        for l_idx, lineage in enumerate(self.lineages):
            curve = self.curves[l_idx]

            # Fit principal curve through data
            # Weights are important as they effectively silence points
            # that are not associated with the lineage.
            curve.fit(
                self.data,
                max_iter=1,
                w=self.cell_weights[:, l_idx]
            )

            if self.debug_plot_lineages:
                cell_mask = np.logical_or.reduce(
                    np.array([self.cluster_labels == k for k in lineage]))
                cells_involved = self.data[cell_mask]
                self.debug_axes[0, 1].scatter(cells_involved[:, 0], cells_involved[:, 1], s=2, alpha=0.5)
                alphas = curve.pseudotimes_interp
                alphas = (alphas - alphas.min()) / (alphas.max() - alphas.min())
                for i in np.random.permutation(self.data.shape[0])[:50]:
                    path_from = (self.data[i][0], curve.points_interp[i][0])
                    path_to = (self.data[i][1], curve.points_interp[i][1])
                    self.debug_axes[0, 1].plot(path_from, path_to, c='black', alpha=alphas[i])
                self.debug_axes[0, 1].plot(curve.points_interp[curve.order, 0],
                                           curve.points_interp[curve.order, 1], label=str(lineage))

            d_sq, dist = curve.project_to_curve(self.data, curve.points_interp[curve.order])
            distances.append(d_sq)
        self.distances = distances
        if self.debug_plot_lineages:
            self.debug_axes[0, 1].legend()

    def calculate_cell_weights(self):
        """TODO: annotate, this is a translation from R"""
        cell_weights = [self.cluster_labels_onehot[:, self.lineages[l].clusters].sum(axis=1)
                        for l in range(len(self.lineages))]
        cell_weights = np.stack(cell_weights, axis=1)

        d_sq = np.stack(self.distances, axis=1)
        d_ord = np.argsort(d_sq, axis=None)
        w_prob = cell_weights/cell_weights.sum(axis=1, keepdims=True)  # shape (cells, lineages)
        w_rnk_d = np.cumsum(w_prob.reshape(-1)[d_ord]) / w_prob.sum()

        z = d_sq
        z_shape = z.shape
        z = z.reshape(-1)
        z[d_ord] = w_rnk_d
        z = z.reshape(z_shape)
        z_prime = 1 - z ** 2
        z_prime[cell_weights == 0] = np.nan
        w0 = cell_weights.copy()
        cell_weights = z_prime / np.nanmax(z_prime, axis=1, keepdims=True) #rowMins(D) / D
        np.nan_to_num(cell_weights, nan=1, copy=False) # handle 0/0
        # cell_weights[is.na(cell_weights)] <- 0
        cell_weights[cell_weights > 1] = 1
        cell_weights[cell_weights < 0] = 0
        cell_weights[w0 == 0] = 0

        reassign = True
        if reassign:
            # add if z < .5
            cell_weights[z < .5] = 1 #(rowMins(D) / D)[idx]

            # drop if z > .9 and cell_weights < .1
            ridx = (z.max(axis=1) > .9) & (cell_weights.min(axis=1) < .1)
            w0 = cell_weights[ridx]
            z0 = z[ridx]
            w0[(z0 > .9) & (w0 < .1)] = 0 # !is.na(Z0) & Z0 > .9 & W0 < .1
            cell_weights[ridx] = w0

        self.cell_weights = cell_weights

    def avg_curves(self):
        """
        Starting at leaves, calculate average curves for each branch

        :return: shrinkage_percentages, cluster_children, cluster_avg_curves
        """
        cell_weights = self.cell_weights
        shrinkage_percentages = list()
        cluster_children = dict()  # maps cluster to children
        lineage_avg_curves = dict()
        cluster_avg_curves = dict()
        branch_clusters = self.branch_clusters.copy()
        if self.debug_level > 0:
            print('Reversing from leaf to root')
        if self.debug_plot_avg:
            self.plotter.clusters(self.debug_axes[1, 0], s=4, alpha=0.4)

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
            if self.debug_level > 0:
                print(f'Averaging branch @{k} with lineages:', branch_lineages, branch_curves)

            avg_curve = self.avg_branch_curves(branch_curves)
            cluster_avg_curves[k] = avg_curve
            # avg.curve$w <- rowSums(vapply(pcurves, function(p){ p$w }, rep(0,nrow(X))))

            # Calculate shrinkage weights using areas where cells share lineages
            # note that this also captures cells in average curves, since the
            # lineages which are averaged are present in branch_lineages
            common = cell_weights[:, branch_lineages] > 0
            common_mask = common.mean(axis=1) == 1.
            shrinkage_percent = dict()
            for curve in branch_curves:
                shrinkage_percent[curve] = self.shrinkage_percent(curve, common_mask)
            shrinkage_percentages.append(shrinkage_percent)

            # Add avg_curve to lineage_avg_curve for cluster_children
            for l in branch_lineages:
                lineage_avg_curves[l] = avg_curve
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
        if self.debug_plot_avg:
            self.debug_axes[1, 0].legend()
        return shrinkage_percentages, cluster_children, cluster_avg_curves

    def shrink_curves(self, cluster_children, shrinkage_percentages, cluster_avg_curves):
        """
        Starting at root, shrink curves for each branch

        Parameters:
            cluster_children:
            shrinkage_percentages:
            cluster_avg_curves:
        :return:
        """
        branch_clusters = self.branch_clusters.copy()
        while len(branch_clusters) > 0:
            # Starting at root, find lineages involves in branch
            k = branch_clusters.popleft()
            shrinkage_percent = shrinkage_percentages.pop()
            branch_curves = list(cluster_children[k])
            cluster_avg_curve = cluster_avg_curves[k]
            if self.debug_level > 0:
                print(f'Shrinking branch @{k} with curves:', branch_curves)

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
                avg = np.interp(#interp1d(
                    s_interp[order],
                    avg_s_interp[avg_order],     # x
                    avg_p_interp[avg_order, j])#,  # y
                    # assume_sorted=True,
                    # bounds_error=False,
                    # fill_value='extrapolate',
                    # extrapolate_extrema=True)
                # avg = lin_interpolator#(s_interp[order])
                shrunk_curve[:, j] = (avg * pct + orig * (1 - pct))
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
            pct_l = np.interp(
                s_interp[order],
                x, y
            )

        return pct_l

    def avg_branch_curves(self, branch_curves):
        """branch_lineages is a list of lineages passing through branch"""
        # s_interps, p_interps, orders
        num_cells = branch_curves[0].points_interp.shape[0]
        num_dims_reduced = branch_curves[0].points_interp.shape[1]

        # 1. Interpolate all the lineages over the shared time domain
        branch_s_interps = np.stack([c.pseudotimes_interp for c in branch_curves], axis=1)
        max_shared_pseudotime = branch_s_interps.max(axis=0).min()  # take minimum of maximum pseudotimes for each lineage
        combined_pseudotime = np.linspace(0, max_shared_pseudotime, num_cells)
        curves_dense = list()
        for curve in branch_curves:
            lineage_curve = np.zeros((combined_pseudotime.shape[0], num_dims_reduced))
            order = curve.order
            # Linearly interpolate each dimension as a function of pseudotime
            for j in range(num_dims_reduced):
                lin_interpolator = interp1d(
                    curve.pseudotimes_interp[order], # x
                    curve.points_interp[order, j],   # y
                    assume_sorted=True
                )
                lineage_curve[:, j] = lin_interpolator(combined_pseudotime)
            curves_dense.append(lineage_curve)

        curves_dense = np.stack(curves_dense, axis=1)  # (n, L_b, J)

        # 2. Average over these curves and project the data onto the result
        avg = curves_dense.mean(axis=1)  # avg is already "sorted"
        avg_curve = PrincipalCurve()
        avg_curve.project_to_curve(self.data, points=avg)
        # avg_curve.pseudotimes_interp -= avg_curve.pseudotimes_interp.min()
        if self.debug_plot_avg:
            self.debug_axes[1, 0].plot(avg[:, 0], avg[:, 1], c='blue', linestyle='--', label='average', alpha=0.7)
            _, p_interp, order = avg_curve.unpack_params()
            self.debug_axes[1, 0].plot(p_interp[order, 0], p_interp[order, 1], c='red', label='data projected', alpha=0.7)

        # avg.curve$w <- rowSums(vapply(pcurves, function(p){ p$w }, rep(0,nrow(X))))
        return avg_curve

    @property
    def unified_pseudotime(self):
        pseudotime = np.zeros_like(self.curves[0].pseudotimes_interp)
        for l_idx, lineage in enumerate(self.lineages):
            curve = self.curves[l_idx]
            cell_mask = np.logical_or.reduce(
                np.array([self.cluster_labels == k for k in lineage]))
            pseudotime[cell_mask] = curve.pseudotimes_interp[cell_mask]
        return pseudotime

    def list_lineages(self, cluster_to_label):
        for lineage in self.lineages:
            print(', '.join([
                cluster_to_label[l] for l in lineage
            ]))
