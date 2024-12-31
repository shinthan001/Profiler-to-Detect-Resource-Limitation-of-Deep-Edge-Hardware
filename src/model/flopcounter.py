import torch
from functools import partial

class FlopCounter:

    def __init__(self):
        pass

    @staticmethod
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

    @staticmethod
    def calculate_conv(module:torch.nn.Module, output:tuple):
        params = FlopCounter.calculate_params(module)
        macs = params * (output.shape[:1] + output.shape[2:]).numel()
        return macs

    @staticmethod
    def calculate_batchnorm(input:torch.tensor)->int:
    # factor 2 indicates subtractions mean and divsion by std.dev
        macs = 2 * input.numel()
        return macs

    @staticmethod
    def calculate_linear(module:torch.nn.Module, output:torch.tensor):
        # Batch * Output (Cout * Wout * Hout)
        output_size = FlopCounter.calculate_params(module)
        macs =   output.shape[0] * output_size
        return macs

    @staticmethod
    def calculate(module:torch.nn.Module, input:torch.tensor, output:torch.tensor):
        macs = 0
        if 'Linear' in module._get_name():
            macs = FlopCounter.calculate_linear(module, output)
        elif 'Conv' in module._get_name():
            macs = FlopCounter.calculate_conv(module,output)
        elif 'Norm' in module._get_name():
            macs = FlopCounter.calculate_batchnorm(input[0])

        return macs

    @staticmethod
    def hook(module, forward_pre_hook=None, forward_hook=None, index=0):
    
        if(len(list(module.children())) == 0):
            if(forward_pre_hook is not None):
                module.register_forward_pre_hook(partial(forward_pre_hook,index)) 
            if(forward_hook is not None):
                module.register_forward_hook(partial(forward_hook, index)) 

        for m in module.children():
            index = FlopCounter.hook(m, forward_pre_hook, forward_hook, index= index+1)
        return index

    @staticmethod
    def profile_layers( model:torch.nn.Module, inputs:torch.tensor):

        module_index, buffer = {}, [0,0]
        
        def forward_hook(index, module, input, output):
            self_flops =  FlopCounter.calculate(module, input, output)
            self_params = FlopCounter.calculate_params(module)
            module_index[index] = [module._get_name(), self_flops, self_params]
            buffer[0] += self_flops
            buffer[1] += self_params

        FlopCounter.hook(model, forward_hook=forward_hook)

        model.eval()

        with torch.no_grad():
            model(inputs)

        module_index['total'] = buffer
        return module_index