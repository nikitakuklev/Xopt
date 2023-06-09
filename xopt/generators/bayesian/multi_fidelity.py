import logging
from copy import deepcopy
from typing import Callable, Dict, Optional

import pandas as pd
import torch
from botorch.acquisition import (
    FixedFeatureAcquisitionFunction,
    GenericMCObjective,
    qUpperConfidenceBound,
)
from pydantic import Field, validator

from xopt.generators.bayesian.custom_botorch.constrained_acqusition import (
    ConstrainedMCAcquisitionFunction,
)
from xopt.generators.bayesian.custom_botorch.multi_fidelity import NMOMF
from xopt.generators.bayesian.mobo import MOBOGenerator
from xopt.vocs import ObjectiveEnum, VOCS

logger = logging.getLogger()


class MultiFidelityGenerator(MOBOGenerator):
    name = "multi_fidelity"
    fidelity_parameter: str = Field(
        "s", description="fidelity parameter name", const=True
    )
    cost_function: Callable = Field(
        lambda x: x + 1.0,
        description="callable function that describes the cost "
        "of evaluating the objective function",
    )
    reference_point: Optional[Dict[str, float]] = None
    supports_multi_objective = True
    supports_batch_generation = True

    __doc__ = """Implements Multi-fidelity Bayeisan optimizationn
        Assumes a fidelity parameter [0,1]
        """

    @validator("vocs", pre=True)
    def validate_vocs(cls, v: VOCS):
        v.variables["s"] = [0, 1]
        v.objectives["s"] = ObjectiveEnum("MAXIMIZE")
        if len(v.constraints):
            raise ValueError(
                "constraints are not currently supported in multi-fidelity BO"
            )

        return v

    def __init__(self, **kwargs):
        super(MultiFidelityGenerator, self).__init__(**kwargs)

        # set reference point
        if self.reference_point is None:
            self.reference_point = {}
            for name, val in self.vocs.objectives.items():
                if name != self.fidelity_parameter:
                    if val == "MAXIMIZE":
                        self.reference_point.update({name: -100.0})
                    else:
                        self.reference_point.update({name: 100.0})
        self.reference_point.update({self.fidelity_parameter: 0.0})

    def calculate_total_cost(self, data: pd.DataFrame = None) -> float:
        """calculate total cost of data samples using the fidelity parameter"""
        if data is None:
            data = self.data

        f_data = self.get_input_data(data)

        # apply callable function to get costs
        return self.cost_function(f_data[..., self.fidelity_variable_index]).sum()

    def _get_acquisition(self, model):
        """
        Creates the Multi-Fidelity Knowledge Gradient acquistion function

        In order for MFKG to evaluate the information gain, it uses the model to
        predict the function value at the highest fidelity after conditioning
        on the observation. This is handled by the project argument, which specifies
        how to transform a tensor X to its target fidelity. We use a default helper
        function called project_to_target_fidelity to achieve this.

        An important point to keep in mind: in the case of standard KG, one can ignore
        the current value and simply optimize the expected maximum posterior mean of the
        next stage. However, for MFKG, since the goal is optimize information gain per
        cost, it is important to first compute the current value (i.e., maximum of the
        posterior mean at the target fidelity). To accomplish this, we use a
        FixedFeatureAcquisitionFunction on top of a PosteriorMean.

        """

        X_baseline = self.get_input_data(self.data)

        # wrap the cost function such that it only has to accept the fidelity parameter
        def true_cost_function(X):
            return self.cost_function(X[..., self.fidelity_variable_index])

        acq_func = NMOMF(
            model=model,
            X_baseline=X_baseline,
            ref_point=self.torch_reference_point,
            cost_call=true_cost_function,
            objective=self._get_objective(),
            constraints=self._get_constraint_callables(),
            cache_root=False,
            prune_baseline=True,
        )

        return acq_func

    def add_data(self, new_data: pd.DataFrame):
        if self.fidelity_parameter not in new_data:
            raise ValueError(
                f"fidelity parameter {self.fidelity_parameter} must be "
                f"in added data"
            )

        # overwrite add data to check for valid fidelity values
        if (new_data[self.fidelity_parameter] > 1.0).any() or (
            new_data[self.fidelity_parameter] < 0.0
        ).any():
            raise ValueError("cannot add fidelity data that is outside the range [0,1]")
        super().add_data(new_data)

    @property
    def fidelity_variable_index(self):
        return self.vocs.variable_names.index(self.fidelity_parameter)

    @property
    def fidelity_objective_index(self):
        return self.vocs.objective_names.index(self.fidelity_parameter)

    def get_optimum(self):
        """select the best point at the maximum fidelity"""

        # define single objective based on vocs
        weights = torch.zeros(self.vocs.n_outputs, **self._tkwargs)
        for idx, ele in enumerate(self.vocs.objective_names):
            if self.vocs.objectives[ele] == "MINIMIZE":
                weights[idx] = -1.0
            elif self.vocs.objectives[ele] == "MAXIMIZE":
                weights[idx] = 1.0

        def obj_callable(Z):
            return torch.matmul(Z, weights.reshape(-1, 1)).squeeze(-1)

        c_posterior_mean = ConstrainedMCAcquisitionFunction(
            self.model,
            qUpperConfidenceBound(
                model=self.model, beta=0.0, objective=GenericMCObjective(obj_callable)
            ),
            self._get_constraint_callables(),
        )

        max_fidelity_c_posterior_mean = FixedFeatureAcquisitionFunction(
            c_posterior_mean,
            self.vocs.n_variables,
            [self.fidelity_variable_index],
            [1.0],
        )

        boundst = self._get_bounds().T
        fixed_bounds = torch.cat(
            (
                boundst[: self.fidelity_variable_index],
                boundst[self.fidelity_variable_index + 1 :],
            )
        ).T

        result = self.numerical_optimizer.optimize(
            max_fidelity_c_posterior_mean, fixed_bounds, 1
        )

        vnames = deepcopy(self.vocs.variable_names)
        del vnames[self.fidelity_variable_index]
        df = pd.DataFrame(result.detach().cpu().numpy(), columns=vnames)
        df[self.fidelity_parameter] = 1.0

        return self.vocs.convert_dataframe_to_inputs(df)
