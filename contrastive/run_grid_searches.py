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
sweeps = [['hx3ax5uc', 45],
          ['sv8hcq00', 45],
          ['ddp9kzke', 45],
          ['2dgc9if9', 45],
          ['l9zec9s3', 45],
          ['lldop02d', 45],
          ['nkmg22jn', 45],
          ['3wgelcay', 45],
          ['e58pz6o5', 45],
          ['oixb9cg0', 45],
          ['pr6zaukq', 45],
          ['f08spq37', 45],
          ['d8e9t80j', 45],
          ['ugwszory', 45],
          ['rbd2giue', 45],
          ['u4s4yraq', 45]]

# max number of agents you want to have in parallel
n_agents = 4


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