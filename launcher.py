import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from hydra.core.hydra_config import HydraConfig
import subprocess
import os
from itertools import product

@hydra.main(version_base=None, config_path="conf", config_name="config")
def launcher(cfg: DictConfig):
    """Launch multiple jobs with parameter sweeps"""
    
    print("Launcher Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Automatically detect sweep parameters (any parameter that is a list)
    sweep_params = {}
    
    for key, value in cfg.items():
        # Skip special keys
        if key in ['available_gpus', 'save_path', 'logging_path', 'cmip_dir', 
                   'ref_dir', 'defaults', 'loss']:
            continue
        
        # If it's a list/tuple with more than 1 element, it's a sweep parameter
        if isinstance(value, (list, tuple, ListConfig)):
            if len(value) > 1 or (len(value) == 1 and isinstance(value[0], (list, tuple))):
                sweep_params[key] = list(value)
                print(f"Detected sweep parameter: {key} = {value}")
            # Single-element lists are treated as fixed values
            elif len(value) == 1:
                print(f"Fixed parameter (single-element list): {key} = {value[0]}")
        # If it's a single value but you explicitly want to sweep it, 
        # make sure it's a list in your config
    
    if not sweep_params:
        print("\nNo sweep parameters detected!")
        print("Make sure your sweep parameters are lists with multiple values in the config.")
        return
    
    # Generate all combinations
    keys = list(sweep_params.keys())
    values = [sweep_params[k] for k in keys]
    combinations = list(product(*values))
    
    print(f"\nTotal combinations: {len(combinations)}")
    print(f"Available GPUs: {cfg.available_gpus}")
    print(f"Sweep parameters: {keys}")
    
    gpu_idx = 0
    num_gpus = len(cfg.available_gpus)
    
    for i, combo in enumerate(combinations, 1):
        # Build override arguments
        overrides = []
        for key, value in zip(keys, combo):
            if isinstance(value, str) and ';' in value:
                overrides.append(f"{key}='{value}'")
            else:
                overrides.append(f"{key}={value}")
        
        # Assign GPU
        gpu = cfg.available_gpus[gpu_idx % num_gpus]
        
        # Create descriptive name for this combination
        combo_name = f"{combo[0]}_deg{combo[1]}_q{combo[2]}"  # e.g., access_cm2_deg2_q0.5
        
        print(f"\n[{i}/{len(combinations)}] Launching job on GPU {gpu}")
        print(f"  Parameters: {dict(zip(keys, combo))}")
        
        # Use descriptive Hydra directory name
        cmd = [
            'python', 'run_exp.py',
            f'hydra.run.dir=hydra_logs/{combo_name}',  # ← Descriptive name
            f'cuda_device={gpu}'
        ] + overrides
        
        print(f"  Command: {' '.join(cmd)}")
        
        subprocess.Popen(cmd, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)})
        
        gpu_idx += 1

if __name__ == "__main__":
    launcher()
