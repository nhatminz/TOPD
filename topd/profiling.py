"""Phase clocks and opt-in bounded CUDA/NCCL operator traces."""
import time
from pathlib import Path

import torch


class PhaseClock:
    """CUDA events avoid synchronizing the host after each training microbatch."""
    def __init__(self, device):
        self.device = device
        self.entries = {}

    def start(self, name):
        event = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
        if event is not None:
            event.record()
        self.entries.setdefault(name, []).append([event, None, time.perf_counter(), None])

    def stop(self, name):
        entry = self.entries[name][-1]
        entry[3] = time.perf_counter()
        if entry[0] is not None:
            entry[1] = torch.cuda.Event(enable_timing=True)
            entry[1].record()

    def seconds(self, name):
        # Caller synchronizes once at the phase/update boundary, never per row.
        return sum(start.elapsed_time(stop)/1000 if start is not None else end-begin
                   for start, stop, begin, end in self.entries.get(name, []))


class UpdateProfiler:
    def __init__(self, settings, step, rank, output_dir, device):
        self.active = step in settings.get("profile_steps", [])
        self.path = Path(output_dir or ".")/"profiles"/f"step-{step:06d}-rank-{rank:05d}.json"
        self.profiler = None
        if self.active:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.profiler = torch.profiler.profile(activities=activities)
            self.profiler.start()

    def finish(self):
        if self.profiler is None:
            return {"communication_profile_status": "not_profiled", "communication_gpu_time_sec": None}
        self.profiler.stop()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.profiler.export_chrome_trace(str(self.path))
        # Kernel duration sum, inclusive of overlaps; never subtract from wall time.
        communication = sum(event.time_range.elapsed_us() for event in self.profiler.events()
            if str(event.device_type).endswith("CUDA") and "nccl" in event.name.lower())/1e6
        return {"communication_profile_status": "profiled_kernel_sum_inclusive_of_overlap",
                "communication_gpu_time_sec": communication, "operator_trace": str(self.path)}
