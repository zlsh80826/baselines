"""Mapping Module

This module implement a mapping game with gym environnment style

"""

import subprocess
import tempfile
import inspect
import logging
import shutil
import random
import signal
import uuid
import time
import os

from gym.envs.registration import register
from gym.spaces import Discrete
from gym.spaces import Box
from gym import Env
import numpy as np

from utils.utils import generate_hostfile, parse_trace


class Environment(Env):
    """
    This environment implement mapping environment
    """
    def __init__(self, log_dir):
        """
        Set the running config
        Set the gym spec
        """
        try:
            num_swaps = int(os.environ['NUM_SWAPS'])
        except:
            assert False, "Not set env 'NUM_SWAPS'"

        self.istrain = bool(os.environ['ISTRAIN'])
        self.create_run_envs()
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            format='%(message)s',
            filename=os.path.join(log_dir, 'runtime-{}'.format(self.uuid)),
            level=logging.INFO)

        self.trace_size = 750
        ob_shape = (self.trace_size, 3)
        self.num_nodes = self.num_jobs = 12
        self.num_swaps = num_swaps
        self.num_train = 1000000
        self.num_eval = 5 
        self.tid = 0 
        self.current_src = self.steps = -1
        self.prev_sim_time = None
        self.current_map = None
        self.elf = '/data/src/latency.smpi'

        self.observation_space = Box(
            low=-np.inf, high=np.inf,
            shape=ob_shape, dtype=np.float32)
        self.action_space = Discrete(self.num_nodes)
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        """
        handle signal int
        """
        print(self.uuid, 'receive signal')
        self.tempdir.cleanup()
        raise NotImplementedError

    def create_run_envs(self):
        """
        setup running environment
        """
        self.uuid = uuid.uuid4()
        print('Environment UUID: {}'.format(self.uuid))
        tempdir = tempfile.TemporaryDirectory()
        self.tempdir = tempdir
        self.trace_file = os.path.join(tempdir.name, f'{self.uuid}.trace')
        self.platform_file = os.path.join(tempdir.name, 'platform.xml')
        self.smpi_opts = ['--cfg=smpi/display-timing:1',
                          '--cfg=smpi/host-speed:2e9']
                          # '--cfg=tracing/smpi/internals:yes']
        self.trace_opts = ['-trace', '-trace-file', self.trace_file]
        self.exe = os.path.join(tempdir.name, 'exe')
        shutil.copyfile('/data/xmldescs/torus2D/12/platform.xml', self.platform_file)
        shutil.copyfile('/data/src/latency.smpi', self.exe)

    def get_cases(self):
        """
        return the testcases base on training mode
        """
        if self.istrain:
            return os.path.join(os.getcwd(), f'/data/testcases/sample-train/{self.tid}.in')
        return os.path.join(os.getcwd(), f'/data/testcases/sample-test/{self.tid}.in')

    def run_simgrid(self):
        """
        run the simgrid simulation in tempdir
        """
        with tempfile.NamedTemporaryFile('w') as temp_file:
            hostfile = generate_hostfile(self.current_map, temp_file)
            cmd = ['smpirun', '-np', str(self.num_jobs), *self.trace_opts,
                   *self.smpi_opts, '-platform', self.platform_file,
                   '-hostfile', hostfile, self.exe, self.get_cases()]
            result = subprocess.run(cmd, cwd=self.tempdir.name, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        sim_time = float([x for x in result.stderr.decode('ascii').splitlines()
                          if 'Simulated time' in x][0].split()[4])
        return sim_time

    def reset(self):
        self.current_map = list(range(self.num_jobs))
        self.current_src = 0
        self.steps = 0

        sim_time = self.run_simgrid()
        self.prev_sim_time = sim_time
        logging.info(f'{self.tid}, {self.current_map}, 0, {sim_time}')

        observations = parse_trace(self.trace_file, self.trace_size)
        return observations

    def step(self, action):
        self.current_map[self.current_src], self.current_map[action] = \
        self.current_map[action], self.current_map[self.current_src]

        sim_time = self.run_simgrid()
        reward = self.prev_sim_time - sim_time
        self.prev_sim_time = sim_time

        self.current_src = (self.current_src + 1) % self.num_jobs
        self.steps = (self.steps + 1) % (self.num_swaps)
        logging.info(f'{self.tid}, {self.current_map}, {reward}, {sim_time}')

        if self.steps == 0:
            if self.istrain:
                self.tid = random.randrange(self.num_train)
                assert self.tid < self.num_train
            else:
                self.tid += 1
                assert self.tid < self.num_eval

        observations = parse_trace(self.trace_file, self.trace_size)
        return observations, reward, self.steps == 0, {}

    def render(self, mode=None):
        pass


register(
    id='Mapping-v0',
    entry_point='mapping:Environment',
)
