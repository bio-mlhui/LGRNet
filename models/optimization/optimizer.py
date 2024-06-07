
from detectron2.solver.build import maybe_add_gradient_clipping
from collections import OrderedDict
from typing import Any, Dict, List, Set, Union, Iterable, Callable, Type, Optional
import copy
import itertools
import torch
from enum import Enum
_GradientClipperInput = Union[torch.Tensor, Iterable[torch.Tensor]]
_GradientClipper = Callable[[_GradientClipperInput], None]


class GradientClipType(Enum):
    VALUE = "value"
    NORM = "norm"


def maybe_add_full_model_gradient_clipping(optim, configs):
    # detectron2 doesn't have full model gradient clipping now
    clip_norm_val = configs['optim']['clip_gradients']['clip_value']
    enable = (
        configs['optim']['clip_gradients']['enabled']
        and configs['optim']['clip_gradients']['clip_type'] == "full_model"
        and configs['optim']['clip_gradients']['clip_value'] > 0.0
    )

    class FullModelGradientClippingOptimizer(optim):
        def step(self, closure=None):
            all_params = itertools.chain(*[x["params"] for x in self.param_groups])
            torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
            super().step(closure=closure)

    return FullModelGradientClippingOptimizer if enable else optim


def _create_gradient_clipper(cfg) -> _GradientClipper:
    """
    Creates gradient clipping closure to clip by value or by norm,
    according to the provided config.
    """
    cfg = copy.deepcopy(cfg)

    def clip_grad_norm(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_norm_(p, cfg['clip_value'], cfg['norm_type'])

    def clip_grad_value(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_value_(p, cfg['clip_value'])

    _GRADIENT_CLIP_TYPE_TO_CLIPPER = {
        GradientClipType.VALUE: clip_grad_value,
        GradientClipType.NORM: clip_grad_norm,
    }
    return _GRADIENT_CLIP_TYPE_TO_CLIPPER[GradientClipType(cfg['clip_type'])]

def _generate_optimizer_class_with_gradient_clipping(
    optimizer: Type[torch.optim.Optimizer],
    *,
    per_param_clipper: Optional[_GradientClipper] = None,
    global_clipper: Optional[_GradientClipper] = None,
) -> Type[torch.optim.Optimizer]:
    """
    Dynamically creates a new type that inherits the type of a given instance
    and overrides the `step` method to add gradient clipping
    """
    assert (
        per_param_clipper is None or global_clipper is None
    ), "Not allowed to use both per-parameter clipping and global clipping"

    def optimizer_wgc_step(self, closure=None):
        if per_param_clipper is not None:
            for group in self.param_groups:
                for p in group["params"]:
                    per_param_clipper(p)
        else:
            # global clipper for future use with detr
            # (https://github.com/facebookresearch/detr/pull/287)
            all_params = itertools.chain(*[g["params"] for g in self.param_groups])
            global_clipper(all_params)
        super(type(self), self).step(closure)

    OptimizerWithGradientClip = type(
        optimizer.__name__ + "WithGradientClip",
        (optimizer,),
        {"step": optimizer_wgc_step},
    )
    return OptimizerWithGradientClip

def maybe_add_gradient_clipping(
    configs: dict, optimizer: torch.optim.Optimizer):
    """
    If gradient clipping is enabled through config options, wraps the existing
    optimizer type to become a new dynamically created class OptimizerWithGradientClip
    that inherits the given optimizer and overrides the `step` method to
    include gradient clipping.

    Args:
        cfg: CfgNode, configuration options
        optimizer: type. A subclass of torch.optim.Optimizer

    Return:
        type: either the input `optimizer` (if gradient clipping is disabled), or
            a subclass of it with gradient clipping included in the `step` method.
    """
    if not configs['optim']['clip_gradients']['enabled']:
        return optimizer
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer_type = type(optimizer)
    else:
        assert issubclass(optimizer, torch.optim.Optimizer), optimizer
        optimizer_type = optimizer

    grad_clipper = _create_gradient_clipper(configs['optim']['clip_gradients'])
    OptimizerWithGradientClip = _generate_optimizer_class_with_gradient_clipping(
        optimizer_type, per_param_clipper=grad_clipper
    )
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer.__class__ = OptimizerWithGradientClip  # a bit hacky, not recommended
        return optimizer
    else:
        return OptimizerWithGradientClip

def get_optimizer(params, configs):
    optimizer_type = configs['optim']['name']
    base_lr = configs['optim']['base_lr']
    weight_decay = configs['optim']['weight_decay'] if 'weight_decay' in configs['optim'] else configs['optim']['base_wd']
    if optimizer_type == "AdamW":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW, configs)(
            params, base_lr, weight_decay=weight_decay,
        )
    else:
        raise NotImplementedError(f"no optimizer type {optimizer_type}")
    
    if configs['optim']['clip_gradients']['clip_type'] != "full_model":
        optimizer = maybe_add_gradient_clipping(configs, optimizer)
    
    return optimizer


