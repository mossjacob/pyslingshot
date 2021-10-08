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
    def __init__(self, data, cluster_labels, start_node=0, debug_level=None):
        self.data = data
        self.cluster_labels_onehot = cluster_labels
        self.cluster_labels = self.cluster_labels_onehot.argmax(axis=1)
        self.num_clusters = self.cluster_labels.max() + 1
        self.start_node = start_node
        cluster_centres = [data[self.cluster_labels == k].mean(axis=0) for k in range(self.num_clusters)]
        self.cluster_centres = np.stack(cluster_centres)
        self.lineages = None      # list of Lineages
        self.prev_curves = None   # list of principle curves len = #lineages
        self.cell_weights = None  # weights indicating cluster assignments
        self.distances = None
        self.branch_clusters = None
        debug_level = 0 if debug_level is None else dict(verbose=1)[debug_level]
        self.debug_level = debug_level
        self.plotter = SlingshotPlotter(self)

        # Construct smoothing kernel for the shrinking step
        self.kernel_x = np.linspace(-3, 3, 512)
        kde = KernelDensity(bandwidth=1., kernel='gaussian')
        kde.fit(np.zeros((self.kernel_x.shape[0], 1)))
        self.kernel_y = np.exp(kde.score_samples(self.kernel_x.reshape(-1, 1)))

    def _set_debug_axes(self, axes):
        self.debug_axes = axes
        self.debug_plot_lineages = axes is not None
        self.debug_plot_avg = axes is not None

    def construct_mst(self, dists, start_node):
        tree = minimum_spanning_tree(dists)

        # Plot MST
        connections = {k: list() for k in range(self.num_clusters)}
        cx = tree.tocoo()
        for i,j,v in zip(cx.row, cx.col, cx.data):
            connections[i].append(j)
            connections[j].append(i)

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
        return children

    def fit(self, num_epochs=10, debug_axes=None):
        self._set_debug_axes(debug_axes)
        if self.prev_curves is None:  # Initial curves and pseudotimes:
            self.get_lineages()
            self.construct_initial_curves()
            self.cell_weights = [self.cluster_labels_onehot[:, self.lineages[l].clusters].sum(axis=1)
                                 for l in range(len(self.lineages))]
            self.cell_weights = np.stack(self.cell_weights, axis=1)

        for epoch in tqdm(range(num_epochs)):
            curves = self.get_curves()

            if epoch == num_epochs - 1:  # plot curves
                self.plotter.clusters(self.debug_axes[1, 1], s=2, alpha=0.5)
                self.plotter.curves(self.debug_axes[1, 1], curves)

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

        self.prev_curves = piecewise_linear
        self.distances = distances

    def get_lineages(self):
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

        tree = self.construct_mst(dists, self.start_node)

        # Plot distance matrix, clusters, and MST
        # from matplotlib import pyplot as plt
        # plt.imshow(dists)
        self.plotter.clusters(self.debug_axes[0, 0])
        for root, children in tree.items():
            for child in children:
                start = [self.cluster_centres[root][0], self.cluster_centres[child][0]]
                end = [self.cluster_centres[root][1], self.cluster_centres[child][1]]
                self.debug_axes[0, 0].plot(start, end, c='black')

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
        if self.debug_level > 0:
            print('Lineages:', lineages)
        self.lineages = lineages
        self.branch_clusters = branch_clusters
        return lineages

    def get_curves(self):
        # Calculate cell weights
        # cell weight is a matrix #cells x #lineages indicating cell-lineage assignment
        self.calculate_cell_weights(self.distances)

        # Fit principal curve for all lineages using existing curves
        curves, cluster_lineages = self.fit_lineage_curves()
        self.debug_axes[0, 1].legend()


        # Ensure starts at 0
        for l_idx, lineage in enumerate(self.lineages):
            min_time = np.min(curves[l_idx].pseudotimes_interp[self.cell_weights[:, l_idx] > 0])
            curves[l_idx].pseudotimes_interp -= min_time

        # Determine average curves
        shrinkage_percentages, cluster_children, cluster_avg_curves =\
            self.avg_curves(curves, cluster_lineages)
        self.debug_axes[1, 0].legend()

        # Shrink towards average curves in areas of cells common to all branch lineages
        self.shrink_curves(cluster_children, shrinkage_percentages, cluster_avg_curves)

        self.prev_curves = curves
        self.debug_plot_lineages = False
        self.debug_plot_avg = False

        return self.prev_curves

    def fit_lineage_curves(self):
        """Updates curve using a cubic spline and projection of data"""
        assert self.lineages is not None
        curves = list()
        cluster_lineages = {k: list() for k in range(self.num_clusters)}

        # Calculate principal curves
        for l_idx, lineage in enumerate(self.lineages):
            # Find cells involved in lineage
            cell_mask = np.logical_or.reduce(
                np.array([self.cluster_labels == k for k in lineage]))
            cells_involved = self.data[cell_mask]
            for k in lineage:
                cluster_lineages[k].append(l_idx)

            prev_curve = self.prev_curves[l_idx]
            curve = prev_curve#PrincipalCurve(k=3)  # cubic
            curve.fit(
                self.data,
                max_iter=1,
                w=self.cell_weights
            )

            from matplotlib import pyplot as plt
            plt.figure()
            # plt.scatter(p[:, 0], p[:, 1], c='red', s=40)
            plt.scatter(curve.points_interp[curve.order][:, 0], curve.points_interp[curve.order][:, 1], c='blue', s=5)

            # curve.fit(self.data, max_iter=1)
            plt.scatter(curve.points_interp[curve.order][:, 0], curve.points_interp[curve.order][:, 1], c='red', s=20, alpha=0.5)

            order = curve.order

            if self.debug_plot_lineages:
                self.debug_axes[0, 1].scatter(cells_involved[:, 0], cells_involved[:, 1], s=2, alpha=0.5)
                alphas = curve.pseudotimes_interp
                alphas = (alphas - alphas.min()) / (alphas.max() - alphas.min())
                for i in np.random.permutation(cells_involved.shape[0])[:50]:
                    path_from = (cells_involved[i][0], curve.points_interp[i][0])
                    path_to = (cells_involved[i][1], curve.points_interp[i][1])
                    self.debug_axes[0, 1].plot(path_from, path_to, c='black', alpha=alphas[i])
                self.debug_axes[0, 1].plot(curve.points_interp[order, 0], curve.points_interp[order, 1], label=str(lineage))
            d_sq, dist = curve.project_to_curve(self.data, curve.points_interp[order])
            curves.append(curve)
        return curves, cluster_lineages

    def calculate_cell_weights(self, distances):
        """TODO: annotate, this is a translation from R"""
        cell_weights = [self.cluster_labels_onehot[:, self.lineages[l].clusters].sum(axis=1)
                        for l in range(len(self.lineages))]
        cell_weights = np.stack(cell_weights, axis=1)

        d_sq = np.stack(distances, axis=1)
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

    def avg_curves(self, curves, cluster_lineages):
        """
        Starting at leaves, calculate average curves for each branch

        :param curves: all lineage curves
        :param cluster_lineages: lineages belonging to each cluster
        :return:
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
            branch_lineages = cluster_lineages[k]
            cluster_children[k] = set()
            for lineage in branch_lineages:  # loop all lineages through branch
                if lineage in lineage_avg_curves:  # add avg curve
                    curve = lineage_avg_curves[lineage]
                else:  # or add leaf curve
                    curve = curves[lineage]
                cluster_children[k].add(curve)

            # Calculate the average curve for this branch
            branch_curves = list(cluster_children[k])
            if self.debug_level > 0:
                print(f'Averaging branch @{k} with lineages:', branch_lineages, branch_curves)
            #branch_lineages, curves
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
        :param curve: tuple (pseudotimes, points)
        :param avg_curve: tuple (pseudotimes, points) for average curve
        :param pct: percentage shrinkage, in same order as curve.pseudotimes
        :return:
        """
        num_cells = branch_curves[0].points_interp.shape[0]
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

        return

    def shrinkage_percent(self, curve, common_ind, method = 'cosine'):
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
            # pct_l = approx(x, y, pts2wt, rule = 2,
            #                 ties = 'ordered').y
            pct_l = np.interp(
                s_interp[order],
                x, y
            )
                # bounds_error=False,
                # fill_value='extrapolate')
            # pct_l = lin_interpolator()

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

        return avg_curve
        #
        #     avg.curve$w <- rowSums(vapply(pcurves, function(p){ p$w }, rep(0,nrow(X))))
        #     return(avg.curve)
        # }