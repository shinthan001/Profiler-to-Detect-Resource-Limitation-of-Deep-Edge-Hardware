import gc
import torch

from tqdm import tqdm
from codecarbon import OfflineEmissionsTracker
from torch.profiler import profile, ProfilerActivity

from profiler.profiler import Profiler


class CUDA(Profiler):

    def __init__(self, track_energy=True, track_flops=False, disable_warmup=False):
        super().__init__()
        self._started = False
        self._alloc_mem, self._rsv_mem = 0, 0
        self._disable_warmup = disable_warmup
        self.disable_print = False

        self._time = 0
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)

        self._kWh = 0
        if(track_energy): self._cc_tracker = OfflineEmissionsTracker(project_name="cuda",
                                                   log_level= 'error', save_to_file=False)
        
        self._flops = 0
        if(track_flops): self._torchprof = profile(activities=[ProfilerActivity.CPU], 
                                        with_flops=True,)
        
        
    def start_profiling(self):
        if(self._started): return
        self._started = True
        if(not self.disable_print): print("Start profiling on CUDA.")
        self.clean_cache()
        if hasattr(self, '_cc_tracker'): self._cc_tracker.start()
        if hasattr(self, '_torchprof'): self._torchprof.start()
        self._start.record()

    def stop_profiling(self):
        if(not self._started): return
        self._alloc_mem = torch.cuda.max_memory_allocated()
        self._rsv_mem = torch.cuda.max_memory_reserved()
        
        self._end.record()
        torch.cuda.synchronize()
        self._time = self._start.elapsed_time(self._end)

        if hasattr(self, '_cc_tracker'):
            self._cc_tracker.stop()
            self._kWh = self._cc_tracker._total_energy.kWh

        if hasattr(self, '_torchprof'): 
            self._torchprof.stop()
            self._flops = self._torchprof.key_averages().total_average().flops

        self._started = False
        self.clean_cache()
        if(not self.disable_print): print("Stopped profiling.")

    def max_alloc_memory(self):
        return self._alloc_mem
    
    def max_rsv_memory(self):
        return self._rsv_mem
    
    def total_time(self):
        return self._time

    def energy(self):
        return self._kWh

    def get_name(self):
        return torch.cuda.get_device_name()

    def clean_cache(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def warmup(self, model, input):
        if(self._disable_warmup): return
        if(not torch.cuda.is_available()):
            print("cuda not found!")
            return
        
        device = torch.device('cuda')        
        input = input.to(device)
        model = model.to(device)

        model.eval()
        with  torch.no_grad():
            for _ in tqdm(range(5), 'CUDA Warm Up..'):
                model(input)

        # del (model, input)
        self.clean_cache()

    def total_flops(self):
        return self._flops