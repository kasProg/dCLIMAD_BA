import itertools
import subprocess
import yaml
import time
import argparse

# -----------------------------
# Parse command-line arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Job launcher for parameter sweep")
parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
args = parser.parse_args()

# -----------------------------
# Load configuration file
# -----------------------------
with open(args.config, 'r') as f:
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
        if isinstance(v, list):
            command += [f"--{k}"] + [str(item) for item in v]
        else:
            command += [f"--{k}", str(v)]

    # Add fixed params
    for k, v in fixed.items():
        if isinstance(v, bool):
            if v:
                command.append(f"--{k}")
        elif isinstance(v, list):
            command += [f"--{k}"] + [str(item) for item in v]
        else:
            command += [f"--{k}", str(v)]

    command += ["--cuda_device", str(gpu_id)]

    print(f"Launching on GPU {gpu_id}: {' '.join(command)}")
    return subprocess.Popen(command)

while param_combinations or any(gpu_jobs.values()):
    # Check for finished jobs
    for gpu_id, job in gpu_jobs.items():
        if job is not None and job.poll() is not None:
            gpu_jobs[gpu_id] = None
        
        # Launch new job if this GPU is free
        if gpu_jobs[gpu_id] is None and param_combinations:
            params = param_combinations.pop(0)
            gpu_jobs[gpu_id] = launch_job(params, gpu_id, run_file)

    # Short sleep to reduce CPU usage
    time.sleep(5)

print("All jobs finished.")
