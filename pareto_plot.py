import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import json

clim = ['access_cm2', 'gfdl_esm4', 'ipsl_cm6a_lr', 'miroc6', 'mpi_esm1_2_lr', 'mri_esm2_0']
colors = plt.cm.tab10(np.linspace(0, 1, len(clim)))  # Use a colormap for distinct colors
markers = ['o', '^', 'v', '<', '>', 'p']  # Circle, triangle up, triangle down, triangle left, triangle right, tri_down
plt.figure(figsize=(10, 6))
for i, c in enumerate(clim):
    exps_path = f'/pscratch/sd/k/kas7897/dCLIMAD_BA/outputs/jobs_lr1e-4/{c}-gridmet/demo_select_gridmet.csv'
    baseline_path = f'/pscratch/sd/k/kas7897/dCLIMAD_BA/outputs/jobs_lr1e-4/{c}-gridmet/baseline_2001_2014.jsonl'
    exps = pd.read_csv(exps_path)

    with open(baseline_path, 'r') as f:
        baseline = [json.loads(line) for line in f]

    
        # Extract R20mm and Rx5day from baseline[0]
    if 'metrics' in baseline[0]:
        r10mm_baseline = baseline[0]['metrics']['R10mm']
        rx1day_baseline = baseline[0]['metrics']['Rx1day']
        r20mm_baseline = baseline[0]['metrics']['R20mm']
        rx5day_baseline = baseline[0]['metrics']['Rx5day']
    else:
        r10mm_baseline = baseline[0]['R10mm']
        rx1day_baseline = baseline[0]['Rx1day']
        r20mm_baseline = baseline[0]['R20mm']
        rx5day_baseline = baseline[0]['Rx5day']


    ## remove rows with r10/r20/rx1/rx5 >100 or <100
    exps = exps[(abs(exps['r10']) < 100) & (abs(exps['r20']) < 100) & (abs(exps['rx1']) < 100) & (abs(exps['rx5']) < 100)]
    # plt.scatter(abs(exps['r20']), abs(exps['rx5']), alpha=0.7, marker=markers[i], color=colors[i], label=c)
    plt.scatter(abs(exps['r10']), abs(exps['rx1']), alpha=0.7, marker=markers[i], color=colors[i], label=c)

    plt.scatter(abs(r10mm_baseline), abs(rx1day_baseline), alpha=0.7, marker=markers[i],
                 edgecolors='black', color=colors[i], linewidths=2, s=120 )

# plt.colorbar(scatter, label='J')
plt.xlabel('R10 (Abs bias%)')
plt.ylabel('Rx1 (Abs bias%)')
plt.title('Scatter plot of R10 vs Rx1')
plt.grid(True)
plt.legend(title="Climate Model (baseline are with solid border)")

plt.savefig('/pscratch/sd/k/kas7897/dCLIMAD_BA/outputs/jobs_lr1e-4/r10_vs_rx1_scatter.png', dpi=300)

k=1