import itertools
import subprocess
import yaml
import time

# Read config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

sweep = config['sweep']
fixed = config['fixed']

# Available GPUs (example: 4 GPUs = [0,1,2,3])
available_gpus = config['available_gpus']
gpu_jobs = {gpu: None for gpu in available_gpus}

# Create all combinations of sweep parameters
keys, values = zip(*sweep.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Total jobs: {len(param_combinations)}")

if config['fixed']['train']:
    run_file = "run_exp.py"
else:
    run_file = "test_exp.py"

def launch_job(params, gpu_id, run_file):
    command = ["python", run_file]
    
    # Add sweep params
    for k, v in params.items():
        command += [f"--{k}", str(v)]
    
    # Add fixed params
    for k, v in fixed.items():
        if isinstance(v, bool):
            if v:
                command.append(f"--{k}")
        else:
            command += [f"--{k}", str(v)]
    
    # Assign GPU dynamically
    command += ["--cuda_device", str(gpu_id)]
    
    print(f"Launching on GPU {gpu_id}: {' '.join(command)}")
    
    return subprocess.Popen(command)

# Launch loop
while param_combinations or any(gpu_jobs.values()):
    # Launch jobs on free GPUs
    for gpu_id, job in gpu_jobs.items():
        if job is None or job.poll() is not None:
            if param_combinations:
                params = param_combinations.pop(0)
                gpu_jobs[gpu_id] = launch_job(params, gpu_id, run_file)
    
    time.sleep(10)  # Wait a little before checking again
