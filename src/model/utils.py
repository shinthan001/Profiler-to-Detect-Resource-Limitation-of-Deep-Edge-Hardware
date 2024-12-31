from functools import partial

import torch

def calculate_params(module:torch.nn.Module):
    
    if(hasattr(module, 'weight_mask') and 
        module.weight_mask is not None):
        return int(module.weight_mask.sum())
    
    weight, bias = 0, 0
    if(hasattr(module, 'bias') and module.bias is not None):
        bias = module.bias.numel()

    if(hasattr(module, 'weight') and module.weight is not None):
        weight = module.weight.numel()
    
    return weight + bias

def hook(module, forward_pre_hook=None, forward_hook=None, index=0):

    if(len(list(module.children())) == 0):
        if(forward_pre_hook is not None):
            module.register_forward_pre_hook(partial(forward_pre_hook,index)) 
        if(forward_hook is not None):
            module.register_forward_hook(partial(forward_hook, index)) 

    for m in module.children():
        index = hook(m, forward_pre_hook, forward_hook, index= index+1)
    return index