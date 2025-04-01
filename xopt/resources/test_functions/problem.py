import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
from botorch.test_functions.base import BaseTestProblem

from xopt.vocs import VOCS

logger = logging.getLogger(__name__)


def make_test_fun_grid(
    ndim: int,
    idx1,
    idx2,
    n1: int,
    n2: int,
    bounds: np.ndarray,
    fixed_values: Optional[dict[int, float]],
):
    ndimbounds = bounds.shape[1]
    assert ndimbounds == ndim
    linspaces = []
    # test_grid_x = np.linspace(*bounds[:, idx1], n1)
    # test_grid_y = np.linspace(*bounds[:, idx2], n2)
    for idx in range(ndim):
        if idx == idx1:
            linspaces.append(np.linspace(*bounds[:, idx1], n1))
        elif idx == idx2:
            linspaces.append(np.linspace(*bounds[:, idx2], n2))
        else:
            assert isinstance(fixed_values[idx], float)
            linspaces.append(np.array([fixed_values[idx]]))
    grids = np.meshgrid(*linspaces, indexing="ij")
    X = np.hstack([g.reshape(-1, 1) for g in grids])
    txx = grids[idx1]
    tyy = grids[idx2]
    return txx, tyy, X


class Problem(ABC):
    name: str = None
    _bounds: list = None
    _start = None
    _optimal_value = None
    _supports_constaints = False
    _default_objective_mode = "MINIMIZE"

    def __init__(
        self,
        n_var: int,
        n_obj: int,
        n_constr=0,
        bounds: tuple | list[tuple] = None,
        var_offset: float | list[float] = None,
    ) -> None:
        self.n_var = n_var
        self.n_obj = n_obj
        self.n_constr = n_constr
        self.constraint = n_constr > 0
        self._bounds = bounds
        self.var_offset = var_offset
        self.obj_scale = 1.0
        if self._bounds is not None:
            for x in self._bounds:
                assert len(x) == 2
                assert x[0] < x[1]

    def _vocs_dict(self):
        vocs_dict = {
            "variables": {f"x{i + 1}": self._bounds[i] for i in range(self.n_var)},
            "objectives": {
                f"y{i + 1}": self._default_objective_mode for i in range(self.n_obj)
            },
        }
        if self.constraint:
            vocs_dict["constraints"] = {
                f"c{i + 1}": ["LESS_THAN", 0.0] for i in range(self.n_constr)
            }
        return vocs_dict

    @property
    def VOCS(self):
        return VOCS(**self._vocs_dict())

    @property
    def bounds(self):
        return self._bounds

    @property
    def bounds_numpy(self):
        # 2 x d
        return np.vstack(self._bounds).T

    @abstractmethod
    def _evaluate(self, x: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def evaluate(self, x: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        assert x.shape[-1] == self.n_var, f"{x.shape[-1]} != {self.n_var}"
        size = x.shape[-1]
        for i in range(size):
            if np.any(x[..., i] > self._bounds[i][1]):
                raise ValueError(f"Input {x} greater than {self._bounds[i][1]}")
            if np.any(x[..., i] < self._bounds[i][0]):
                raise ValueError(f"Input {x} lower than {self._bounds[i][0]}")
        shifted_x = x - self.var_offset
        raw_y = self._evaluate(shifted_x, **kwargs)
        transformed_y = self.scale * self._evaluate(shifted_x, **kwargs)
        return

    def evaluate_dict(self, inputs: dict, *args, **kwargs) -> dict[str, float]:
        ind = np.array([inputs[f"x{i + 1}"] for i in range(self.n_var)])
        obj, cs = self.evaluate(ind[None, :], **kwargs)
        assert obj.shape == (1, self.n_obj), f"Bad {obj.shape=} {self.n_obj=}"
        outputs = {}
        for i in range(self.n_obj):
            outputs[f"y{i + 1}"] = obj[0, i].item()
        if self.constraint:
            assert cs.shape == (1, self.n_constr), f"Bad {cs.shape=} {self.n_constr=}"
            for i in range(self.n_constr):
                outputs[f"c{i + 1}"] = cs[0, i].item()
        return outputs

    @property
    def optimal_value(self) -> float:
        return self._optimal_value

    def sample_on_grid(self, n=None, lower=None, upper=None):
        linspaces = []
        for i in range(self.n_var):
            if lower is None or upper is None:
                linspaces.append(np.linspace(self._bounds[i][0], self._bounds[i][1], n))
            else:
                linspaces.append(np.linspace(lower, upper, n))
        grids = np.meshgrid(*linspaces, indexing="ij")
        test_x = np.stack([g.reshape(-1, 1) for g in grids], -1)
        logger.debug(f"Sampling test function with {test_x.shape=}")
        result = self._evaluate(test_x)
        r0 = [result[0][:, i].reshape(n, n) for i in range(result[0].shape[1])]
        r1 = (
            [result[1][:, i].reshape(n, n) for i in range(result[1].shape[1])]
            if result[1] is not None
            else None
        )
        return grids, (r0, r1)

    def make_grid(self, n, idx1=None, idx2=None, fixed_values=None):
        idx1 = idx1 or 0
        idx2 = idx2 or 1
        return make_test_fun_grid(
            self.n_var, idx1, idx2, n, n, self.bounds_numpy, fixed_values
        )

    def eval_on_grid(
        self, n, idx1=None, idx2=None, fixed_values=None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, list]:
        txx, tyy, X = self.make_grid(n, idx1, idx2, fixed_values)
        val, c = self.evaluate(X)
        obj_grids = []
        for i in range(val.shape[-1]):
            obj_grids.append(val[:, i].reshape(txx.shape))
        if c is not None:
            c_grids = []
            for i in range(c.shape[-1]):
                c_grids.append(c[:, i].reshape(txx.shape))
        else:
            c_grids = None
        return txx, tyy, X, obj_grids, c_grids


class ConstrainedProblem(Problem, ABC):
    pass


class BTProblem(Problem):
    parent: BaseTestProblem

    @abstractmethod
    def _evaluate(self, x: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @property
    def optimal_value(self) -> float:
        return self.parent.optimal_value

    @property
    def optimizers(self):
        return self.parent._optimizers


class MOProblem(Problem, ABC):
    _max_hv: float = None

    @property
    @abstractmethod
    def ref_point(self) -> np.ndarray:
        pass

    @property
    def max_hv(self):
        return self._max_hv

    @property
    def ref_point_dict(self):
        rp = self.ref_point
        return {f"y{i + 1}": rp[i] for i in range(self.n_obj)}

    # def evaluate_dict(self, inputs: Dict, *args, **params):
    #     ind = np.array([inputs[f"x{i + 1}"] for i in range(self.n_var)])
    #     if ind.ndim == 1:
    #         # MOBO yields floats
    #         ind = ind[np.newaxis, :]
    #     else:
    #         # Random generator yields length-1 numpy arrays
    #         ind = ind.T
    #     objectives, constraints = self.evaluate(ind)
    #     outputs = {}
    #     for i in range(self.n_obj):
    #         outputs[f"y{i + 1}"] = objectives[0, i].item()
    #     for i in range(self.n_constr):
    #         outputs[f"c{i + 1}"] = constraints[0, i].item()
    #     return outputs


class MOBTProblem(MOProblem, BTProblem):
    @property
    def max_hv(self):
        return self.parent._max_hv
