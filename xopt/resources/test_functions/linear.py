import numpy as np

from xopt import VOCS
from xopt.resources.test_functions.problem import MOProblem, Problem


class Linear(Problem):
    name = "linear"
    _supports_constraints = True

    def __init__(
        self,
        n_var: int = 3,
        var_offset: float | list[float] = None,
        obj_scale: float = 1.0,
        obj_offset: float = 0.7,
        bounds: tuple | list[tuple] = (-10.0, 10.0),
        constraints=None,
        negate=False,
    ) -> None:
        if constraints is not None:
            if isinstance(constraints, tuple):
                assert len(constraints) == 1
                self._constraints = constraints
            else:
                self._constraints = (3.0,)

        super().__init__(
            n_var,
            n_obj=1,
            n_constr=1 if self._constraints else 0,
            bounds=[bounds for _ in range(n_var)],
            var_offset=var_offset,
        )

        self.scale = obj_scale
        self.var_offset = var_offset
        self.negate = negate
        self.obj_offset = obj_offset
        self.vocs = self.VOCS

    @property
    def _optimal_value(self):
        if self.negate:
            return (
                0.0 + self.obj_offset if not self.constraint else -3.0 + self.obj_offset
            )
        else:
            return (
                0.0 + self.obj_offset if not self.constraint else 3.0 + self.obj_offset
            )

    @property
    def optimizers(self):
        return [[self.obj_offset] * self.n_var]

    @property
    def VOCS(self):
        vocs_dict = {
            "variables": {
                f"x{i + 1}": list(self._bounds[i]) for i in range(self.n_var)
            },
            "objectives": {"y1": "MINIMIZE" if not self.negate else "MAXIMIZE"},
        }
        if self.constraint:
            if self.negate:
                vocs_dict["constraints"] = {"c1": ["LESS_THAN", -3.0 + self.obj_offset]}
            else:
                vocs_dict["constraints"] = {
                    "c1": ["GREATER_THAN", 3.0 + self.obj_offset]
                }
        return VOCS(**vocs_dict)

    def _evaluate(self, x, *args, **kwargs):
        assert x.shape[-1] == self.n_var
        # objective = self.scale * np.linalg.norm(x - self.offset, axis=-1, keepdims=True)
        objective = self.scale * np.sum(
            np.abs(x - self.obj_offset), axis=-1, keepdims=True
        )
        objective = objective if not self.negate else -objective
        objective = objective + self.obj_offset
        if self.constraint:
            return objective, objective
        else:
            return objective, None


class LinearMO(MOProblem):
    name = "mo_linear"

    def __init__(self, n_var=8, scale=1.0, offset=2.5, negate=False):
        super().__init__(n_var, n_obj=2)
        self.scale = scale
        self.offset = offset
        self.negate = negate
        self._bounds = [(0, 6.0) for _ in range(n_var)]
        self._default_objective_mode = "MAXIMIZE" if self.negate else "MINIMIZE"
        self.shift = 0.0
        self.vocs = self.VOCS

    @property
    def ref_point(self):
        rp = 5.0
        rp = np.sqrt(self.n_var * rp**2)
        if self.shift:
            rp += self.shift
        rp_array = np.array([rp, rp])
        # print(f"Linear ref point: {rp_array}")
        return rp_array

    def _evaluate(self, x, *args, **kwargs):
        assert x.shape[-1] == self.n_var
        objective1 = self.scale * np.linalg.norm(
            x - self.offset, axis=-1, keepdims=True
        )
        objective2 = self.scale * np.linalg.norm(x, axis=-1, keepdims=True)
        objective = np.hstack([objective1.reshape(-1, 1), objective2.reshape(-1, 1)])
        objective = objective if not self.negate else -objective
        return objective, None
