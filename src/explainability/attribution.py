from captum.attr import FeatureAblation
from typing import Any, Callable, Tuple, Union
import torch
from captum._utils.common import _select_targets, _format_inputs, _format_additional_forward_args
from captum._utils.typing import TargetType
from torch import Tensor
from inspect import signature
import pandas as pd
import numpy as np


def compute_gradients(
    forward_fn: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    target_ind: TargetType = None,
    additional_forward_args: Any = None,
) -> Tuple[Tensor, ...]:
    r"""
    Computes gradients of the output with respect to inputs for an
    arbitrary forward function.

    Args:

        forward_fn: forward function. This can be for example model's
                    forward function.
        input:      Input at which gradients are evaluated,
                    will be passed to forward_fn.
        target_ind: Index of the target class for which gradients
                    must be computed (classification only).
        additional_forward_args: Additional input arguments that forward
                    function requires. It takes an empty tuple (no additional
                    arguments) if no additional arguments are required
    """
    with torch.autograd.set_grad_enabled(True):
        # runs forward pass
        outputs = _run_forward(forward_fn, inputs, target_ind, additional_forward_args)
        assert outputs[0].numel() == 1, (
            "Target not provided when necessary, cannot"
            " take gradient with respect to multiple outputs."
        )
        # torch.unbind(forward_out) is a list of scalar tensor tuples and
        # contains batch_size * #steps elements
        grads = torch.autograd.grad(torch.unbind(outputs), inputs)
    return grads


def _run_forward(
    forward_func: Callable,
    inputs: Any,
    target: TargetType = None,
    additional_forward_args: Any = None,
) -> Tensor:
    forward_func_args = signature(forward_func).parameters
    if len(forward_func_args) == 0:
        output = forward_func()
        return output if target is None else _select_targets(output, target)

    # make everything a tuple so that it is easy to unpack without
    # using if-statements
    inputs = _format_inputs(inputs)
    additional_forward_args = _format_additional_forward_args(additional_forward_args)

    output = forward_func(
        (*inputs, *additional_forward_args)
        if additional_forward_args is not None
        else inputs
    )
    return _select_targets(output, target)


class FeatureAblationV2(FeatureAblation):

    def _strict_run_forward(self, *args, **kwargs) -> Tensor:
        """
        A temp wrapper for global _run_forward util to force forward output
        type assertion & conversion.
        Remove after the strict logic is supported by all attr classes
        """
        forward_output = _run_forward(*args, **kwargs)
        if isinstance(forward_output, Tensor):
            return forward_output
    
        output_type = type(forward_output)
        assert output_type is int or output_type is float, (
            "the return of forward_func must be a tensor, int, or float,"
            f" received: {forward_output}"
        )
    
        # using python built-in type as torch dtype
        # int -> torch.int64, float -> torch.float64
        # ref: https://github.com/pytorch/pytorch/pull/21215
        return torch.tensor(forward_output, dtype=output_type)


def plot_attribution_algorithm_comparison(features, algorithms, names, weights, top_n = 10, figsize = (20, 8)):
    df, most_important_features = compute_most_important_features_based_attribution(features=features, algorithms=algorithms, names=names, top_n =top_n)
    weights = torch.cat(weights, dim= 1).mean(0).detach().numpy()
    df["Weights"] = weights / np.linalg.norm(weights, ord=1)
    df = df.loc[most_important_features]
    _ = df.plot(kind= "bar", figsize = figsize, ylabel= "Attributions", xlabel= "Feature", 
                title= "Comparing input feature importances across multiple algorithms and learned weights")


def compute_most_important_features_based_attribution(features, algorithms, names, top_n = 10):
    df = pd.DataFrame([], index = features)
    for name,alg in zip(names, algorithms):
        attr_sum = torch.cat(alg, dim= 1).detach().numpy().sum(0)
        df[name] = attr_sum / np.linalg.norm(attr_sum, ord=1)
    most_important_features = df[names].abs().mean(1).sort_values(ascending= False)[:top_n].index
    return df, most_important_features

