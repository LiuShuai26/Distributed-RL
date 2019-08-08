import numpy as np
import os
import sys
import gym
import datetime
from spinup.utils.run_utils import setup_logger_kwargs


# TODO Parameters_sac1
class ParametersSac1:
    def __init__(self, env_name, total_epochs, num_workers=1):
        # parameters set

        # ray_servr_address = ""

        # self.env_name = 'LunarLanderContinuous-v2'   # 'MountainCarContinuous-v0'
        self.env_name = env_name
        # BipedalWalker-v2
        # Pendulum-v0
        # self.env_name = 'MountainCarContinuous-v0'

        # TODO
        env = gym.make(env_name)
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = env.action_space.high[0]

        self.ac_kwargs = dict(hidden_sizes=[400, 300])
        # Share information about action space with policy architecture
        self.action_space = env.action_space
        self.ac_kwargs['action_space'] = env.action_space

        self.total_epochs = total_epochs
        self.num_workers = num_workers

        self.alpha = 0.1

        self.gamma = 0.99
        self.replay_size = 1000000
        self.lr = 1e-3
        self.polyak = 0.995

        self.steps_per_epoch = 5000
        self.batch_size = 100
        self.start_steps = 10000
        self.max_ep_len = 1000
        self.save_freq = 1

        self.seed = 0

        self.train = True
        self.continue_training = False

        exp_name = "dsac1_" + self.env_name + "_workers_num=" + str(self.num_workers) + "_" + str(datetime.datetime.now())

        self.logger_kwargs = setup_logger_kwargs(exp_name, self.seed)

        self.summary_dir = './tboard_ddpg'  # Directory for storing tensorboard summary results
        self.save_dir = './model_ddpg'      # Directory for storing trained model

        self.parameter_servers = ["localhost:2222"]
        self.workers = []
        for i in range(num_workers):
            self.workers.append("localhost:"+str(2223+i))

    def get_opt(self):
        return self
