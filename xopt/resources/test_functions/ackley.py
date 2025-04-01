import numpy as np

from xopt.resources.test_functions.problem import Problem
from xopt.vocs import VOCS

variables = {f"x{i}": [-5, 10] for i in range(20)}
objectives = {"y": "MINIMIZE"}

vocs = VOCS(variables=variables, objectives=objectives)


class Ackley(Problem):
    name = "ackley"
    _optimal_value = 0.0
    _default_objective_mode = "MINIMIZE"

    def __init__(
            self,
            n_var: int = 2,
            a: float = 20.0,
            b: float = 1.0 / 5,
            c: float = 2 * np.pi,
            var_offset: float | list[float] = None,
    ):
        bounds = [(-32.768, 32.768) for i in range(n_var)]
        super().__init__(n_var=n_var, n_obj=1, bounds=bounds, var_offset=var_offset)
        self.a = a
        self.b = b
        self.c = c
        self._start = [32.768 / 2 for i in range(n_var)]

    def _evaluate(self, x, *args, **kwargs):
        part1 = (
                -1.0
                * self.a
                * np.exp(
                -1.0 * self.b * np.sqrt((1.0 / self.n_var) * np.sum(x * x, axis=-1))
        )
        )
        part2 = -1.0 * np.exp((1.0 / self.n_var) * np.sum(np.cos(self.c * x), axis=-1))
        objective = part1 + part2 + self.a + np.exp(1)
        return objective, None


default_problem = Ackley(n_var=20)


def ackley(x):
    return default_problem.evaluate(x)


def evaluate_ackley_np(inputs: dict):
    return {
        "y": default_problem.evaluate(np.array([inputs[k] for k in sorted(inputs)]))
    }


def evaluate_ackley(inputs: dict):
    x = np.array([inputs[f"x{i}"] for i in range(20)])
    return {"y": default_problem.evaluate(x)}
