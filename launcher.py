import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from hydra.core.hydra_config import HydraConfig
import subprocess
import os
from itertools import product

@hydra.main(version_base=None, config_path="config_hydra", config_name="config")
def launcher(cfg: DictConfig):
    """Launch multiple jobs with parameter sweeps"""
    
    print("="*80)
    print("Launcher Configuration:")
    print("="*80)
    print(OmegaConf.to_yaml(cfg))
    print("="*80)
    
    # Debug: Print all top-level keys
    print("\nTop-level config keys:")
    print(list(cfg.keys()))
    print()
    
    # Automatically detect sweep parameters (any parameter that is a list with >1 element)
    sweep_params = {}
    
    # Keys to skip (these should not be sweep parameters even if they're lists)
    skip_keys = {
        'available_gpus', 'save_path', 'logging_path', 'cmip_dir', 
        'ref_dir', 'defaults', 'loss', 'spatial_extent', 'spatial_extent_val'
    }
    
    for key, value in cfg.items():
        # Skip special keys
        if key in skip_keys:
            continue
        
        # If it's a list/tuple with more than 1 element, it's a sweep parameter
        if isinstance(value, (list, tuple, ListConfig)):
            if len(value) > 1:
                sweep_params[key] = list(value)
                print(f"✓ Detected sweep parameter: {key} = {value}")
            # Single-element lists are treated as fixed values
            elif len(value) == 1:
                print(f"  Fixed parameter (single-element list): {key} = {value[0]}")
        else:
            # Print scalar values for debugging
            print(f"  Scalar parameter: {key} = {value}")
    
    if not sweep_params:

        return
    
    # Generate all combinations
    keys = list(sweep_params.keys())
    values = [sweep_params[k] for k in keys]
    combinations = list(product(*values))
    
    print("\n" + "="*80)
    print("Sweep Summary:")
    print("="*80)
    print(f"Total combinations: {len(combinations)}")
    print(f"Available GPUs: {cfg.available_gpus}")
    print(f"Sweep parameters: {keys}")
    print("="*80 + "\n")
    
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
        
        # Create descriptive name for this combination (safely)
        try:
            # Try to create a descriptive name
            combo_dict = dict(zip(keys, combo))
            combo_name = f"{combo_dict.get('clim', 'run')}_deg{combo_dict.get('degree', 'X')}_q{combo_dict.get('emph_quantile', 'X')}"
        except:
            combo_name = f"combo_{i}"
        
        print(f"\n[{i}/{len(combinations)}] Launching job on GPU {gpu}")
        print(f"  Parameters: {dict(zip(keys, combo))}")
        
        # Launch command (removed hydra.run.dir as discussed)
        cmd = ['python', 'run_exp.py'] + overrides
        
        print(f"  Command: {' '.join(cmd)}")
        
        subprocess.Popen(cmd, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)})
        
        gpu_idx += 1

if __name__ == "__main__":
    launcher()
