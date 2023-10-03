"""
This program is meant to automatize the process of launching wandb sweeps, which allows
to run the grid search on all the regions with one command line. However, the sweeps have
to made beforehand, which is pretty long and tedious.
"""

import os
from multiprocessing import Pool

# You have to create the sweeps before using this script.
# Then, you have to fill the following list with [sweep_id, count], count being the number 
# of models you want to train in the given sweep.
sweeps = [['vsaqwun7', 15],
          ['zq69nsgf', 15],
          ['fid18gtx', 15],
          ['a1k10z70', 15],
          ['w54ys71y', 15],
          ['i6pb0us8', 15],
          ['qwaex7kj', 15]]

# max number of agents you want to have in parallel
n_agents = 2


def launch_one_sweep(sweep):
    """Launches a sweep and closes it when it is over.
    Arguments:
        - sweep: a list containing the sweep id and the number of models to train."""
    sweep_id, count = sweep
    os.system(f'wandb agent aymeric-gaudin/2023_brain_regions_grid_searches/{sweep_id} --count {count}')
    os.system(f'wandb sweep --stop aymeric-gaudin/2023_brain_regions_grid_searches/{sweep_id}')

if __name__ == '__main__':
    # start n_agents worker processes
    with Pool(processes=n_agents) as pool:

        pool.map(launch_one_sweep, sweeps)

    # exiting the 'with'-block has stopped the pool
    print("Now the pool is closed and no longer available")