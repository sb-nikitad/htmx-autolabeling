import psutil
import pynvml
import json
import time

# Initialize the NVML library
pynvml.nvmlInit()

# Replace 1234 with the PID of the process you want to monitor
process = psutil.Process(19931)

# Create empty dictionaries to store the peak usage for each resource
cpu_peak = {}
gpu_peak = {}
ram_peak = {}
storage_peak = {}

# Monitor the process for 60 seconds with a 1-second interval
print('Monitoring process')
   # Get CPU usage
for i in range(600):
    cpu_percent = process.cpu_percent()
    cpu_bytes = process.cpu_times().system + process.cpu_times().user
    if cpu_bytes > cpu_peak.get('bytes', 0):
        cpu_peak['bytes'] = cpu_bytes
        cpu_peak['time'] = i
    
    # Get GPU usage (assuming Nvidia GPU)
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # assuming one GPU is installed
    gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    if gpu_utilization > gpu_peak.get('percent', 0):
        gpu_peak['percent'] = gpu_utilization
        gpu_peak['memory'] = gpu_memory
        gpu_peak['time'] = i
    
    # Get RAM usage
    ram_percent = process.memory_percent()
    ram_bytes = process.memory_info().rss
    ram_gb = ram_bytes / (1024**3)
    if ram_gb > ram_peak.get('gb', 0):
        ram_peak['gb'] = ram_gb
        ram_peak['time'] = i
    
    # Get storage usage
    storage_bytes = process.io_counters().read_bytes + process.io_counters().write_bytes
    storage_mb = storage_bytes / (1024**2)
    if storage_mb > storage_peak.get('mb', 0):
        storage_peak['mb'] = storage_mb
        storage_peak['time'] = i
    
    # Print usage to console
    print('--------------------------------')
    print("CPU usage:", cpu_percent, "%")
    print("GPU usage:", gpu_memory/(1024**3), "GB of GPU memory and", gpu_utilization, "percent of GPU utilization")
    print("RAM usage:", ram_gb, "GB")
    print("Storage usage:", storage_mb, "MB")

    # Sleep for 1 second
    time.sleep(1)
    
# Print peak usage to console
print("Peak CPU usage:", cpu_peak['bytes'], "bytes at time", cpu_peak['time'], "seconds")
print("Peak GPU usage:", gpu_peak['memory']/(1024**3), "GB of GPU memory and", gpu_peak['percent'], "percent of GPU utilization at time", gpu_peak['time'], "seconds")
print("Peak RAM usage:", ram_peak['gb'], "GB at time", ram_peak['time'], "seconds")
print("Peak storage usage:", storage_peak['mb'], "MB at time", storage_peak['time'], "seconds")

# Write peak usage to stats.json
stats = {
    'CPU': cpu_peak,
    'GPU': gpu_peak,
    'RAM': ram_peak,
    'Storage': storage_peak
}

with open('stats.json', 'w') as f:
    json.dump(stats, f, indent=4)

# Release the NVML library
pynvml.nvmlShutdown()
