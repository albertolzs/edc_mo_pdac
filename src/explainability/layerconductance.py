import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Any, Callable, List, Tuple, Union, Dict
from captum._utils.typing import TargetType
from torch import device, Tensor
from captum.attr._utils.approximation_methods import approximation_parameters
from captum.attr._utils.common import _reshape_and_sum
from captum._utils.common import (_expand_additional_forward_args, _expand_target, _format_additional_forward_args, _format_output, _format_inputs, _select_targets, _sort_key_list, _reduce_list)
from captum._utils.gradient import _forward_layer_distributed_eval, _extract_device_ids, apply_gradient_requirements
from captum._utils.typing import ModuleOrModuleList
from torch.nn import Module
from inspect import signature
from collections import defaultdict
import threading
import torch


def _attribute(self, inputs: Tuple[Tensor, ...], baselines: Tuple[Union[Tensor, int, float], ...], target: TargetType = None, additional_forward_args: Any = None, n_steps: int = 50,
               method: str = "gausslegendre", attribute_to_layer_input: bool = False, 
               step_sizes_and_alphas: Union[None, Tuple[List[float], List[float]]] = None,) -> Union[Tensor, Tuple[Tensor, ...]]:
    num_examples = inputs[0].shape[0]
    if step_sizes_and_alphas is None:
        # Retrieve scaling factors for specified approximation method
        step_sizes_func, alphas_func = approximation_parameters(method)
        alphas = alphas_func(n_steps + 1)
    else:
        _, alphas = step_sizes_and_alphas
    # Compute scaled inputs from baseline to final input.
    scaled_features_tpl = tuple(
        torch.cat([baseline + alpha * (input - baseline) for alpha in alphas], dim=0).requires_grad_()
        for input, baseline in zip(inputs, baselines)
    )

    additional_forward_args = _format_additional_forward_args(
        additional_forward_args
    )
    # apply number of steps to additional forward args
    # currently, number of steps is applied only to additional forward arguments
    # that are nd-tensors. It is assumed that the first dimension is
    # the number of batches.
    # dim -> (#examples * #steps x additional_forward_args[0].shape[1:], ...)
    input_additional_args = (
        _expand_additional_forward_args(additional_forward_args, n_steps + 1)
        if additional_forward_args is not None
        else None
    )
    expanded_target = _expand_target(target, n_steps + 1)

    # Conductance Gradients - Returns gradient of output with respect to
    # hidden layer and hidden layer evaluated at each input.
    (layer_gradients, layer_evals,) = compute_layer_gradients_and_eval(
        forward_fn=self.forward_func,
        layer=self.layer,
        inputs=scaled_features_tpl,
        additional_forward_args=input_additional_args,
        target_ind=expanded_target,
        device_ids=self.device_ids,
        attribute_to_layer_input=attribute_to_layer_input,
    )

    # Compute differences between consecutive evaluations of layer_eval.
    # This approximates the total input gradient of each step multiplied
    # by the step size.
    grad_diffs = tuple(
        layer_eval[num_examples:] - layer_eval[:-num_examples]
        for layer_eval in layer_evals
    )

    # Element-wise multiply gradient of output with respect to hidden layer
    # and summed gradients with respect to input (chain rule) and sum
    # across stepped inputs.
    attributions = tuple(
        _reshape_and_sum(
            grad_diff * layer_gradient[:-num_examples],
            n_steps,
            num_examples,
            layer_eval.shape[1:],
        )
        for layer_gradient, layer_eval, grad_diff in zip(
            layer_gradients, layer_evals, grad_diffs
        )
    )
    return _format_output(len(attributions) > 1, attributions)


def compute_layer_gradients_and_eval(forward_fn: Callable, layer: ModuleOrModuleList, inputs: Union[Tensor, Tuple[Tensor, ...]], target_ind: TargetType = None,
    additional_forward_args: Any = None, gradient_neuron_selector: Union[None, int, Tuple[Union[int, slice], ...], Callable] = None, device_ids: Union[None, List[int]] = None,
    attribute_to_layer_input: bool = False, output_fn: Union[None, Callable] = None,) -> Union[
    Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]],
    Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...], Tuple[Tensor, ...]],
    Tuple[List[Tuple[Tensor, ...]], List[Tuple[Tensor, ...]]],]:
    with torch.autograd.set_grad_enabled(True):
        # saved_layer is a dictionary mapping device to a tuple of
        # layer evaluations on that device.
        saved_layer, output = _forward_layer_distributed_eval(
            forward_fn,
            inputs,
            layer,
            target_ind=target_ind,
            additional_forward_args=additional_forward_args,
            attribute_to_layer_input=attribute_to_layer_input,
            forward_hook_with_return=True,
            require_layer_grads=True,
        )
        
        assert output[0].numel() == 1, (
            "Target not provided when necessary, cannot"
            " take gradient with respect to multiple outputs."
        )

        device_ids = _extract_device_ids(forward_fn, saved_layer, device_ids)

        # Identifies correct device ordering based on device ids.
        # key_list is a list of devices in appropriate ordering for concatenation.
        # If only one key exists (standard model), key list simply has one element.
        key_list = _sort_key_list(
            list(next(iter(saved_layer.values())).keys()), device_ids
        )
        all_outputs: Union[Tuple[Tensor, ...], List[Tuple[Tensor, ...]]]
        if isinstance(layer, Module):
            all_outputs = _reduce_list(
                [
                    saved_layer[layer][device_id]
                    if output_fn is None
                    else output_fn(saved_layer[layer][device_id])
                    for device_id in key_list
                ]
            )
        else:
            all_outputs = [
                _reduce_list(
                    [
                        saved_layer[single_layer][device_id]
                        if output_fn is None
                        else output_fn(saved_layer[single_layer][device_id])
                        for device_id in key_list
                    ]
                )
                for single_layer in layer
            ]
        all_layers: List[Module] = [layer] if isinstance(layer, Module) else layer
        grad_inputs = tuple(
            layer_tensor
            for single_layer in all_layers
            for device_id in key_list
            for layer_tensor in saved_layer[single_layer][device_id]
        )
        saved_grads = torch.autograd.grad(torch.unbind(output), grad_inputs)

        offset = 0
        all_grads: List[Tuple[Tensor, ...]] = []
        for single_layer in all_layers:
            num_tensors = len(next(iter(saved_layer[single_layer].values())))
            curr_saved_grads = [
                saved_grads[i : i + num_tensors]
                for i in range(
                    offset, offset + len(key_list) * num_tensors, num_tensors
                )
            ]
            offset += len(key_list) * num_tensors
            if output_fn is not None:
                curr_saved_grads = [
                    output_fn(curr_saved_grad) for curr_saved_grad in curr_saved_grads
                ]

            all_grads.append(_reduce_list(curr_saved_grads))

        layer_grads: Union[Tuple[Tensor, ...], List[Tuple[Tensor, ...]]]
        layer_grads = all_grads
        if isinstance(layer, Module):
            layer_grads = all_grads[0]

        if gradient_neuron_selector is not None:
            assert isinstance(
                layer, Module
            ), "Cannot compute neuron gradients for multiple layers simultaneously!"
            inp_grads = _neuron_gradients(
                inputs, saved_layer[layer], key_list, gradient_neuron_selector
            )
            return (
                cast(Tuple[Tensor, ...], layer_grads),
                cast(Tuple[Tensor, ...], all_outputs),
                inp_grads,
            )
    return layer_grads, all_outputs  # type: ignore


def _forward_layer_distributed_eval(
    forward_fn: Callable,
    inputs: Any,
    layer: ModuleOrModuleList,
    target_ind: TargetType = None,
    additional_forward_args: Any = None,
    attribute_to_layer_input: bool = False,
    forward_hook_with_return: bool = False,
    require_layer_grads: bool = False,
) -> Union[
    Tuple[Dict[Module, Dict[device, Tuple[Tensor, ...]]], Tensor],
    Dict[Module, Dict[device, Tuple[Tensor, ...]]],
]:
    r"""
    A helper function that allows to set a hook on model's `layer`, run the forward
    pass and returns intermediate layer results, stored in a dictionary,
    and optionally also the output of the forward function. The keys in the
    dictionary are the device ids and the values are corresponding intermediate layer
    results, either the inputs or the outputs of the layer depending on whether we set
    `attribute_to_layer_input` to True or False.
    This is especially useful when we execute forward pass in a distributed setting,
    using `DataParallel`s for example.
    """
    saved_layer: Dict[Module, Dict[device, Tuple[Tensor, ...]]] = defaultdict(dict)
    lock = threading.Lock()
    all_layers: List[Module] = [layer] if isinstance(layer, Module) else layer

    # Set a forward hook on specified module and run forward pass to
    # get layer output tensor(s).
    # For DataParallel models, each partition adds entry to dictionary
    # with key as device and value as corresponding Tensor.
    def hook_wrapper(original_module):
        def forward_hook(module, inp, out=None):
            eval_tsrs = inp if attribute_to_layer_input else out
            is_eval_tuple = isinstance(eval_tsrs, tuple)

            if not is_eval_tuple:
                eval_tsrs = (eval_tsrs,)
            if require_layer_grads:
                apply_gradient_requirements(eval_tsrs, warn=False)
            with lock:
                nonlocal saved_layer
                # Note that cloning behaviour of `eval_tsr` is different
                # when `forward_hook_with_return` is set to True. This is because
                # otherwise `backward()` on the last output layer won't execute.
                if forward_hook_with_return:
                    saved_layer[original_module][eval_tsrs[0].device] = eval_tsrs
                    eval_tsrs_to_return = tuple(
                        eval_tsr.clone() for eval_tsr in eval_tsrs
                    )
                    if not is_eval_tuple:
                        eval_tsrs_to_return = eval_tsrs_to_return[0]
                    return eval_tsrs_to_return
                else:
                    saved_layer[original_module][eval_tsrs[0].device] = tuple(
                        eval_tsr.clone() for eval_tsr in eval_tsrs
                    )

        return forward_hook

    all_hooks = []
    try:
        for single_layer in all_layers:
            if attribute_to_layer_input:
                all_hooks.append(
                    single_layer.register_forward_pre_hook(hook_wrapper(single_layer))
                )
            else:
                all_hooks.append(
                    single_layer.register_forward_hook(hook_wrapper(single_layer))
                )
        output = _run_forward(
            forward_fn,
            inputs,
            target=target_ind,
            additional_forward_args=additional_forward_args,
        )
    finally:
        for hook in all_hooks:
            hook.remove()

    if len(saved_layer) == 0:
        raise AssertionError("Forward hook did not obtain any outputs for given layer")

    if forward_hook_with_return:
        return saved_layer, output
    return saved_layer


def _run_forward(forward_func: Callable, inputs: Any, target: TargetType = None, additional_forward_args: Any = None,) -> Tensor:
    forward_func_args = signature(forward_func).parameters
    if len(forward_func_args) == 0:
        output = forward_func()
        return output if target is None else _select_targets(output, target)

    # make everything a tuple so that it is easy to unpack without
    # using if-statements
    inputs = _format_inputs(inputs)
    additional_forward_args = _format_additional_forward_args(additional_forward_args)

    output = forward_func(inputs)
    return _select_targets(output, target)


def plot_comparison_attributions_weights(lc_attr, weights, layer_name = "embedding", figsize=(15, 8)):
    plt.figure(figsize=figsize)
    x_axis_data = np.arange(lc_attr.shape[1])
    y_axis_lc_attr = lc_attr.mean(0)
    y_axis_lc_attr = y_axis_lc_attr / np.linalg.norm(y_axis_lc_attr, ord=1)
    
    y_axis_layer_weight = weights.mean(1)
    y_axis_layer_weight = y_axis_layer_weight / np.linalg.norm(y_axis_layer_weight, ord=1)
    width = 0.25
    legends = ['Attributions','Weights']
    ax = plt.subplot()
    ax.set_title(f'Aggregated neuron importances and learned weights in the {layer_name} layer of the model')
    ax.bar(x_axis_data + width, y_axis_lc_attr, width, align='center', alpha=0.5, color='red')
    ax.bar(x_axis_data + 2 * width, y_axis_layer_weight, width, align='center', alpha=0.5, color='green')
    plt.legend(legends, prop={'size': 15})
    ax.autoscale_view()
    if len(y_axis_layer_weight) <= 100:
        x_axis_labels = [f'Neuron {i}' for i in range(len(y_axis_layer_weight))]
        ax.set_xticks(x_axis_data + 0.5)
        ax.set_xticklabels(x_axis_labels, rotation = "vertical")
    plt.show()


def plot_attribution_distribution(cond_vals, strong_features, weak_features, figsize = (10, 4)):
    cond_vals = pd.DataFrame(cond_vals).iloc[:, strong_features + weak_features]
    cond_vals.plot(kind = "hist", subplots=True, layout=(2,len(strong_features)), figsize = figsize, bins = 100, style="blue", 
                   title = "Distribution of each neuron's weights", sharex = True,  sharey = True)