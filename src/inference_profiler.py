import os
import argparse

import torch

from profiler.cpu import CPU
from profiler.cuda import CUDA
from dataset.imagenet import get_transforms
from utils.utils import append_csv, create_raw_PIL_images

os.environ.update({'KINETO_LOG_LEVEL' : '99'})
DEFAULT_LOG_PATH = "records.csv"

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--batch', type=int, required=True)
parser.add_argument('--width', type=int, required=True)
parser.add_argument('--height', type=int, required=True)
parser.add_argument('--log', type=str, required=False)
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()


if __name__ == '__main__':

    model_path = args.model_path
    batch_size = args.batch
    input_w, input_h = args.width, args.height
    log_path = DEFAULT_LOG_PATH if args.log is None else args.log
    use_cpu = args.cpu

    print(f"\nmodel path: {model_path}\n"+
          f"batch size: {batch_size}| "+
          f"input shape: ({input_w}, {input_h})| "+
          f"use cpu : {use_cpu}| \n" +
          f"log path: {log_path}\n")

    # select device
    device = torch.device('cpu')
    if(torch.cuda.is_available() and not use_cpu):
        device = torch.device('cuda')

    # invoke profiler
    prof = CUDA() if(device.type == 'cuda') else CPU()
    
    # profile static memory of model, don't move this code.
    # always put at top
    prof.start_profiling()
    model = torch.load(model_path)
    model = model.to(device)
    prof.stop_profiling()

    # static memory size of model
    model_alloc_size = prof.max_alloc_memory()
    model_rsv_size = prof.max_rsv_memory()
    print(f"Model Size: {model_alloc_size/1024**2:.2f} MB")
    model.eval()

    transform = get_transforms((input_w, input_h))    
    # create raw tensors
    inputs = create_raw_PIL_images(batch_size, input_w, input_h)    
    # profile transfomring
    prof.start_profiling()
    inputs = [transform(i) for i in inputs]
    inputs = torch.stack(inputs, dim=0).to(device)
    prof.stop_profiling()
    transform_time = prof.total_time()
    print(f"\nCrated input shape: {inputs.shape}\n" +
          f"Transform + Transfer time : {transform_time:.3f} ms\n")

    # warmup for cuda    
    if(hasattr(prof,'warmup')): prof.warmup(model, inputs)

    # inference profiling
    with torch.no_grad():
        prof.start_profiling()
        outputs = model(inputs)
        prof.stop_profiling()

    # statistics
    max_alloc_mem = prof.max_alloc_memory()
    max_rsv_mem = prof.max_rsv_memory()
    total_time = prof.total_time()
    energy =  prof.energy()    
    device_name = prof.get_name()

    print(f"\nDevice Name: {device_name}\n" +
          f"Max Allocated Memory: {max_alloc_mem/1024**2:.2f} MB |" +
          f"Max Reserved Memory: {max_rsv_mem/1024**2:.2f} MB |" +
          f"Total Time: {total_time:.3f} ms |" +
          f"Energy: {energy:.5f} kWh |")

    # save data
    append_csv(log_path, [model_path, model_alloc_size, model_rsv_size, batch_size, (input_w,input_h), 
                          max_alloc_mem, max_rsv_mem, transform_time, total_time, energy, device_name])