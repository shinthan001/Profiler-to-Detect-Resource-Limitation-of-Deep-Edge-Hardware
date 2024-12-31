
import gc
import platform

import torch
from codecarbon import OfflineEmissionsTracker
from torch.profiler import profile, ProfilerActivity

from profiler.profiler import Profiler

class CPU(Profiler):

    def __init__(self, track_energy=True, track_flops=False):
        super().__init__()
        self._torchprof = profile(activities=[ProfilerActivity.CPU], 
                                        profile_memory=True,)
        if(track_flops): self._torchprof.with_flops = True

        # set default number of threads        
        n_threads = torch.get_num_threads()
        print(f"Number of CPU threads: {n_threads}")
        torch.set_num_threads(n_threads)

        self._kWh = 0
        if(track_energy): self._cc_tracker = OfflineEmissionsTracker(project_name="cuda",
                                                    log_level= 'error', save_to_file=False)
        self._started = False
        self.disable_print = False


    def start_profiling(self):
        if(self._started): return
        self._started = True
        if(not self.disable_print): print("Starting Torch Profiler on CPU.")
        if hasattr(self, '_cc_tracker'): self._cc_tracker.start()
        self._torchprof.start()

    def stop_profiling(self):
        if(not self._started): return
        self._started = False
        self._torchprof.stop()
        if(not self.disable_print): print("Stopped Torch Profiler.")
        if hasattr(self, '_cc_tracker'): 
            self._cc_tracker.stop()
            self._kWh = self._cc_tracker._total_energy.kWh
        gc.collect()

    def max_alloc_memory(self):
        if(self.check_empty_event()): return 0
        return max([e.cpu_memory_usage for e in self._torchprof.events()])
    
    # TODO: need to find out a way to retrieve reserved memory by differnt OS
    def max_rsv_memory(self):
        if(self.check_empty_event()): return 0
        return 0
    
    def total_time(self):
        if(self.check_empty_event()): return 0
        return self._torchprof.key_averages().self_cpu_time_total / 10**3 #ms

    def energy(self):
        return self._kWh

    def get_name(self):
        try:
            # For Linux
            with open("/proc/cpuinfo", "r") as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("model"):
                        return line.split(":")[1].strip()
            return None
        except:
            try:
                return platform.processor() # For Windows
            except:
                return None
            
    def check_empty_event(self):
        if(len(self._torchprof.events()) == 0): 
            print("0 event was recorded.")
            return True
        return False
    
    def total_flops(self):
        return self._torchprof.key_averages().total_average().flops