import numpy as np

from scipy.interpolate import interp1d as ScipyInterp1d
from numpy import asarray


class interp1d(ScipyInterp1d):
    def __init__(self, x, y, extrapolate_extrema=True,
                 kind='linear', bounds_error=None, fill_value=np.nan,
                 assume_sorted=False, **kwargs):
        super().__init__(x, y,
                         kind=kind,
                         bounds_error=bounds_error,
                         assume_sorted=assume_sorted,
                         fill_value=fill_value,
                         **kwargs)
        self._bound_extrema = extrapolate_extrema

    def _evaluate(self, x_new):
        # 1. Handle values in x_new that are outside of x. Throw error,
        #    or return a list of mask array indicating the outofbounds values.
        #    The behavior is set by the bounds_error variable.
        x_new = asarray(x_new)
        y_new = self._call(self, x_new)
        below_bounds, above_bounds = self._check_bounds(x_new)
        if not self._extrapolate:
            if len(y_new) > 0:
                # Note fill_value must be broadcast up to the proper size
                # and flattened to work here
                y_new[below_bounds] = self._fill_value_below
                y_new[above_bounds] = self._fill_value_above
        else:
            if self._bound_extrema:
                y_new[below_bounds] = self.y.min()
                y_new[above_bounds] = self.y.max()
        return y_new
