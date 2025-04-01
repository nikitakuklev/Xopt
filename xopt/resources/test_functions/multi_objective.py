import math
from abc import abstractmethod
from typing import Dict

import numpy as np
from scipy.special import gamma

from xopt.resources.test_functions.problem import MOProblem, Problem

from xopt.vocs import VOCS


# From BoTorch
class DTLZ2(MOProblem):
    def __init__(self, n_var=3, negate=False):
        # negate -> maximization
        super().__init__(n_var, n_obj=2)
        self.negate = negate
        self._bounds = [(0.0, 1.0) for _ in range(n_var)]
        self.vocs = self.VOCS

    @property
    def VOCS(self):
        op = "MAXIMIZE" if self.negate else "MINIMIZE"
        vocs_dict = {
            "objectives": {"y1": op, "y2": op},
            "variables": {f"x{i + 1}": self._bounds[i] for i in range(self.n_var)},
        }
        return VOCS(**vocs_dict)

    @property
    def ref_point(self):
        return np.ones(self.n_var) * 1.1

    @property
    def _max_hv(self) -> float:
        # hypercube - volume of hypersphere in R^d such that all coordinates are
        # positive
        hypercube_vol = 1.1**self.n_obj
        pos_hypersphere_vol = (
            math.pi ** (self.n_obj / 2) / gamma(self.n_obj / 2 + 1) / 2**self.n_obj
        )
        return hypercube_vol - pos_hypersphere_vol

    def _evaluate(self, X: np.ndarray, **kwargs) -> np.ndarray:
        assert X.shape[1] == self.n_var
        k = X.shape[1] - 2 + 1
        X_m = X[..., -k:]
        g_X = ((X_m - 0.5) ** 2).sum(axis=-1)
        g_X_plus1 = 1 + g_X
        fs = []
        pi_over_2 = np.pi / 2
        for i in range(self.n_obj):
            idx = 2 - 1 - i
            f_i = g_X_plus1.copy()
            f_i *= np.cos(X[..., :idx] * pi_over_2).prod(axis=-1)
            if i > 0:
                f_i *= np.sin(X[..., idx] * pi_over_2)
            fs.append(f_i)
        return np.stack(fs, axis=-1), None
