import os
import argparse

import torch

from profiler.cpu import CPU
from profiler.cuda import CUDA
from model.tree import TreeNode
# from model.flopcounter import FlopCounter
from model.utils import calculate_params, hook
from utils.utils import blockPrint, enablePrint

os.environ.update({'KINETO_LOG_LEVEL' : '99'})

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--batch', type=int, required=True)
parser.add_argument('--width', type=int, required=True)
parser.add_argument('--height', type=int, required=True)
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--level', type=int, required=False, default=3)

args = parser.parse_args()

if __name__ == '__main__':

    model_path = args.model_path
    batch_size = args.batch
    input_w, input_h = args.width, args.height
    use_cpu = args.cpu
    level = args.level

    print(f"\nmodel path: {model_path}\n"+
          f"batch size: {batch_size}| "+
          f"input shape: ({input_w}, {input_h})| "+
          f"Level: {level}|\n")
   
    # select device
    device = torch.device('cpu')
    if(torch.cuda.is_available() and not use_cpu):
        device = torch.device('cuda')

    # invoke profiler
    prof = CUDA(track_energy=False, 
                track_flops=True) if(device.type == 'cuda') else CPU(track_energy=False, track_flops=True)


    model = torch.load(model_path)
    model = model.to(device)
    model.eval()

    # create dummy inputs
    inputs = torch.rand(batch_size, 3, input_w, input_h, dtype=torch.float32)
    inputs = inputs.to(device)

    # warmup for cuda    
    if(hasattr(prof,'warmup')): prof.warmup(model, inputs)

    module_index, buffer = {}, [0,0]
    
    def forward_pre_hook(index, module, input):
        prof.start_profiling()

    def forward_hook(index, module, input, output):
        prof.stop_profiling()
        # self_flops =  FlopCounter.calculate(module, input, output)
        # self_params = FlopCounter.calculate_params(module)
        self_flops = prof.total_flops()
        self_params = calculate_params(module)
        module_index[index] = [module._get_name(), self_flops, self_params,
                                prof.max_alloc_memory(), prof.max_rsv_memory(),prof.total_time()]
        buffer[0] += self_flops
        buffer[1] += self_params

    # hooking model
    # FlopCounter.hook(model, forward_pre_hook=forward_pre_hook, forward_hook=forward_hook)
    hook(model, forward_pre_hook=forward_pre_hook, forward_hook=forward_hook)

    # inference
    blockPrint()
    with torch.no_grad():
        model(inputs)
    enablePrint()

    buffer.append(prof.get_name())
    module_index['total'] = buffer

    print(f"\nFLOPs: {module_index['total'][0]/1024**3:.2f} G\n" +
          f"Params: {module_index['total'][1]/1024**2:.2f} M\n" +
          f"Device: {module_index['total'][2]}")
    
    print("\n[Allocated Memory | Reserved Memory | Inference Time | FLOPs]")
    root = TreeNode(model._get_name())
    TreeNode.add_tree_nodes(root, model, module_index)
    TreeNode.print_tree(root, level=level)