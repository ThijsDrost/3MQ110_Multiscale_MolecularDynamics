import time
import os
import numpy as np
from Simulator_new import Simulation


dataloc = r'D:\Uni\Master\Mulitscale_methods\MD\Code\Data'

def data_loc(end: str):
    return rf'{dataloc}\{end}'


seeds = [57843920, 7340921, 8973447, 8950432, 43278204, 5629209, 37489219, 74328790]
existing_files = os.listdir(dataloc)
save_points = 1e5
run_time = 100
particles = 50
samples_per_power = 5
power_lims = (2, 6)
samples = samples_per_power*(power_lims[1] - power_lims[0]) + 1  # 26
steps = np.logspace(2, 6, samples, dtype=int)
save_every_arr = (steps / save_points).astype(int)
save_every_arr[save_every_arr < 1] = 1
log_steps = np.log10(steps)

# %%
for index, seed in enumerate(seeds):
    for verlet_type in ['euler', 'velocity', 'basic']:
        start = time.time()
        for step, save_every in zip(steps, save_every_arr):
            # if f'5_deltaE_v{verlet_type}_t{run_time}_p{particles}_s{step}_ns{seed}.npz' in existing_files:
            #     continue
            print(f'{index}, {step}, {save_every}')
            sim = Simulation.run(run_time, step, particles, verlet_type=verlet_type, save_every=save_every, seed=seed)
            sim.save(data_loc(f'5_deltaE_v{verlet_type}_t{run_time}_p{particles}_s{step}_ns{seed}.npz'))
        print(f'{verlet_type} done after {time.time()-start:.0f}s ')
