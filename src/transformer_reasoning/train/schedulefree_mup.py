from collections import defaultdict
import torch
from schedulefree import AdamWScheduleFree, AdamWScheduleFreeClosure

def process_param_groups(params, **kwargs):
    param_groups = list(params)
    if not isinstance(param_groups[0], dict):
        param_groups = [{"params": param_groups}]
    for param_group in param_groups:
        if "lr" not in param_group:
            param_group["lr"] = kwargs["lr"]
        if "weight_decay" not in param_group:
            param_group["weight_decay"] = kwargs.get("weight_decay", 0.)
    return param_groups

def MuAdamW_ScheduleFree(params, impl=AdamWScheduleFree, decoupled_wd=False, **kwargs):
    """Adam with μP scaling.

    Note for this to work properly, your model needs to have its base shapes set
    already using `mup.set_base_shapes`.
    
    Inputs:
        impl: the specific Adam-like optimizer implementation from torch.optim or
            elsewhere 
        decoupled_wd: if True, skips the mup scaling for weight decay, which should
            be used for optimizer implementations that decouple weight decay from
            learning rate. See https://github.com/microsoft/mup/issues/1 for a use case.
    Outputs:
        An instance of `impl` with refined parameter groups, each of which has the correctly
        scaled learning rate according to mup.
    """
    new_param_groups = []
    for param_group in process_param_groups(params, **kwargs):
        # For every existing param group, we split into several new groups
        def new_group():
            new_g = {k:v for k, v in param_group.items() if k != "params"}
            new_g["params"] = []
            return new_g
        # The matrix-like weights might need multiple groups since weights
        # might have different width multipliers
        matrix_like_p = defaultdict(new_group) # key is width_mult
        vector_like_p = new_group()
        for p in param_group["params"]:
            assert hasattr(p, "infshape"), (
                f"A parameter with shape {p.shape} does not have `infshape` attribute. "
                "Did you forget to call `mup.set_base_shapes` on the model?")
            if p.infshape.ninf() == 2:
                matrix_like_p[p.infshape.width_mult()]["params"].append(p)
            elif p.infshape.ninf() > 2:
                raise NotImplementedError("more than 2 inf dimensions")
            else:
                vector_like_p["params"].append(p)
        for width_mult, group in matrix_like_p.items():
            # Scale learning rate and weight decay accordingly
            group["lr"] /= width_mult
            if not decoupled_wd:
                group["weight_decay"] *= width_mult
        new_param_groups.extend(list(matrix_like_p.values()) + [vector_like_p])
    return impl(new_param_groups, **kwargs)