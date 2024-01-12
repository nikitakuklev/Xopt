import logging
import time
from typing import Dict

import pandas as pd
import torch
from botorch.acquisition import FixedFeatureAcquisitionFunction, InverseCostWeightedUtility
from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement, qHypervolumeKnowledgeGradient
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import _get_hv_value_function
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.models.cost import FixedCostModel
from botorch.optim import optimize_acqf
from pydantic import Field

from xopt.generators.bayesian.bayesian_generator import MultiObjectiveBayesianGenerator

from xopt.generators.bayesian.objectives import create_mobo_objective

logger = logging.getLogger(__name__)


class MOBOGenerator(MultiObjectiveBayesianGenerator):
    name = "mobo"
    __doc__ = """Implements Multi-Objective Bayesian Optimization using the Expected
            Hypervolume Improvement acquisition function"""

    def _get_objective(self):
        return create_mobo_objective(self.vocs, self._tkwargs)

    def get_acquisition(self, model):
        """
        Returns a function that can be used to evaluate the acquisition function
        """
        if model is None:
            raise ValueError("model cannot be None")

        # get base acquisition function
        acq = self._get_acquisition(model)

        # apply fixed features if specified in the generator
        if self.fixed_features is not None:
            # get input dim
            dim = len(self.model_input_names)
            columns = []
            values = []
            for name, value in self.fixed_features.items():
                columns += [self.model_input_names.index(name)]
                values += [value]

            acq = FixedFeatureAcquisitionFunction(
                acq_function=acq, d=dim, columns=columns, values=values
            )

        return acq

    def _get_acquisition(self, model):
        inputs = self.get_input_data(self.data)
        sampler = self._get_sampler(model)

        if self.log_transform_acquisition_function:
            acqclass = qLogNoisyExpectedHypervolumeImprovement
        else:
            acqclass = qNoisyExpectedHypervolumeImprovement

        acq = acqclass(
            model,
            X_baseline=inputs,
            constraints=self._get_constraint_callables(),
            ref_point=self.torch_reference_point,
            sampler=sampler,
            objective=self._get_objective(),
            cache_root=False,
            prune_baseline=True,
        )

        return acq


class DMOBOGenerator(MOBOGenerator):
    name = "dmobo"
    objective_costs: Dict[str, float] = Field(..., description="Cost for each objective")
    num_fantasies = 8
    num_pareto = 10
    __doc__ = """Implements Decoupled Multi-Objective Bayesian Optimization using the Expected
            Hypervolume Improvement acquisition function"""

    def _get_objective(self):
        return create_mobo_objective(self.vocs, self._tkwargs)

    @property
    def cost_model(self):
        objective_costs = self.objective_costs
        objective_costs = {o: objective_costs[o] for o in self.vocs.objective_names}
        objective_costs_t = torch.tensor([objective_costs[k] for k in sorted(objective_costs.keys())], **self._tkwargs)
        cost_model = FixedCostModel(fixed_cost=objective_costs_t)
        return cost_model

    def generate(self, n_candidates: int) -> pd.DataFrame:
        self.n_candidates = n_candidates
        if n_candidates > 1 and not self.supports_batch_generation:
            raise NotImplementedError("Only 1 candidate can be generated at a time")

        if self.data is None or self.data.empty:
            raise RuntimeError("No data available to generate candidates")

        timing_results = {}

        # update internal model with internal data
        start_time = time.perf_counter()
        model = self.train_model(self.data)
        timing_results["training"] = time.perf_counter() - start_time

        start_time = time.perf_counter()
        candidates, obj_indices = self.propose_candidates(model, n_candidates=n_candidates)
        logger.debug(f"DMOBO: picked objective {obj_indices}")
        timing_results["acquisition_optimization"] = time.perf_counter() - start_time

        # post process candidates
        result = self._process_candidates(candidates)

        if self.computation_time is not None:
            self.computation_time = pd.concat(
                (
                    self.computation_time,
                    pd.DataFrame(timing_results, index=[0]),
                ),
                ignore_index=True,
            )
        else:
            self.computation_time = pd.DataFrame(timing_results, index=[0])

        # return self.vocs.convert_numpy_to_inputs(candidates.detach().numpy())
        result["__objective_idx__"] = obj_indices[0]
        return result

    def propose_candidates(self, model, n_candidates=1):
        # calculate optimization bounds
        bounds = self._get_optimization_bounds()

        # get acquisition function
        acq_funct = self.get_acquisition(model)

        # get candidates
        candidates, obj_indices = self.numerical_optimizer.optimize(acq_funct, bounds, n_candidates)
        return candidates, obj_indices

    def get_current_value(self, model, ref_point, bounds):
        """
        Helper to get the hypervolume of the current hypervolume maximizing set.
        """
        curr_val_acqf = _get_hv_value_function(
            model=model,
            ref_point=ref_point,
            use_posterior_mean=True,
        )
        _, current_value = optimize_acqf(
            acq_function=curr_val_acqf,
            bounds=bounds,
            q=self.num_pareto,
            num_restarts=10,
            raw_samples=512,
            return_best_only=True,
            options={"batch_limit": 5},
        )
        return current_value

    def _get_acquisition(self, model):
        # objective_costs = self.objective_costs
        # objective_costs = {o: objective_costs[o] for o in self.vocs.objective_names}
        # objective_costs_t = torch.tensor(
        #     [objective_costs[k] for k in sorted(objective_costs.keys())], **self._tkwargs
        # )

        cost_model = self.cost_model
        cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

        bounds = self._get_optimization_bounds()

        current_value = self.get_current_value(
            model=model,
            ref_point=self.torch_reference_point,
            bounds=bounds,
        )

        acq_func = qHypervolumeKnowledgeGradient(
            model=model,
            ref_point=self.torch_reference_point,
            num_fantasies=self.num_fantasies,
            num_pareto=self.num_pareto,
            current_value=current_value,
            cost_aware_utility=cost_aware_utility,
        )

        return acq_func
