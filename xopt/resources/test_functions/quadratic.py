import numpy as np

from xopt import VOCS
from xopt.resources.test_functions.problem import MOProblem, Problem


class Quadratic(Problem):
    name = "quadratic"
    _optimal_value = 0.0
    _supports_constraints = True

    def __init__(
        self, n_var=2, scale=1.0, var_offset=0.2, constraint=False, negate=False
    ) -> None:
        super().__init__(n_var, n_obj=1, n_constr=1 if constraint else 0,
                         bounds=[(-3.0, 3.0) for _ in range(n_var)],
                         var_offset=var_offset)
        self.scale = scale
        self.negate = negate
        self._default_objective_mode = "MAXIMIZE" if self.negate else "MINIMIZE"
        self.vocs = self.VOCS

    def _vocs_dict(self):
        vocs_dict = super().VOCS
        if self.constraint:
            vocs_dict["constraints"] = {"c1": ["GREATER_THAN", 3]}
        return vocs_dict

    def _evaluate(self, x, *args, **kwargs):
        assert x.shape[-1] == self.n_var
        objective = (
            self.scale * np.linalg.norm(x, axis=-1, keepdims=True) ** 2
        )
        objective = objective if not self.negate else -objective
        if self.constraint:
            return objective, x.sum(axis=1, keepdims=True)
        else:
            return objective, None


class QuadraticMO(MOProblem):
    """Quadratic multi-objective test problem - by default, finding minima with 1 objective offset"""

    _ref_point = np.array([5.0, 5.0])

    def __init__(self, n_var=3, scale=1.0, offset=1.5, negate=False):
        # negate -> maximization
        super().__init__(n_var, n_obj=2)
        self.scale = scale
        self.offset = offset
        self.negate = negate
        self._bounds = [(0, 3.0) for _ in range(n_var)]
        self.vocs = self.VOCS
        self.shift = 0

    @property
    def ref_point(self):
        rp = self._ref_point
        rp = rp**self.n_var
        if self.shift:
            rp += self.shift
        return rp

    @property
    def VOCS(self):
        op = "MAXIMIZE" if self.negate else "MINIMIZE"
        vocs_dict = {
            "objectives": {"y1": op, "y2": op},
            "variables": {f"x{i + 1}": self._bounds[i] for i in range(self.n_var)},
        }
        return VOCS(**vocs_dict)

    def _evaluate(self, x, *args, **kwargs):
        # Keep objectives roughly at unit magnitude
        assert x.shape[-1] == self.n_var
        dim_factor = (1.0**2) ** self.n_var
        scale = self.scale / dim_factor
        objective1 = (
            scale * np.linalg.norm(x - self.offset, axis=-1, keepdims=True) ** 2
        )
        objective2 = scale * np.linalg.norm(x, axis=-1, keepdims=True) ** 2
        objective = np.hstack([objective1.reshape(-1, 1), objective2.reshape(-1, 1)])
        objective = objective if not self.negate else -objective
        return objective, None
