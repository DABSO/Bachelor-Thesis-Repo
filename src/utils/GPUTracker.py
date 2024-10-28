import asyncio
import subprocess
import time

class GPUUsageTracker:
    def __init__(self, interval=5):
        self.interval = interval
        self.gpu_usage = {}

    async def get_gpu_usage(self):
        output = await asyncio.create_subprocess_shell(
            'nvidia-smi --query-gpu=utilization.gpu --format=csv,nounits,noheader',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await output.communicate()
        gpu_usage = stdout.decode().strip().split('\n')
        return [int(util) for util in gpu_usage]

    async def track_gpu_usage(self):
        while True:
            timestamp = int(time.time())
            usage = await self.get_gpu_usage()
            self.gpu_usage[timestamp] = usage
            await asyncio.sleep(self.interval)

    def start_tracking(self):
        asyncio.create_task(self.track_gpu_usage())
