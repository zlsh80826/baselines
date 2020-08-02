"""
utils for mapping
"""

import subprocess
import numpy as np

def generate_hostfile(map_, file):
    """
    generate hostfile base on map_
    """
    for node in map_:
        print('node-{}.simgrid.org'.format(node), file=file)
    file.flush()
    return file.name


def parse_trace(filename, trace_size):
    """
    parse the MPI trace
    """
    states = list()
    result = subprocess.run(['pj_dump', filename], stdout=subprocess.PIPE, check=True)
    for line in result.stdout.decode('ascii').splitlines():
        if line.endswith('PMPI_Ssend'):
            _, rank, _, start, end, _, _, _ = line.replace('\n', '').split(',')
            rank = rank[6:]
            states.append(list(map(float, [start, end, rank])))
    while len(states) < trace_size:
        states.append([0., 0., -1])
    slices = len(states) // trace_size
    return np.asarray(states[::slices][:trace_size])
