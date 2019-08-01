import numpy as np
import os
import sys
import datetime
from spinup.utils.run_utils import setup_logger_kwargs


class Parameters:
    def __init__(self, workers_num=1):
        # parameters set

        # self.env_name = 'LunarLanderContinuous-v2'   # 'MountainCarContinuous-v0'
        self.env_name = 'Pendulum-v0'
        # BipedalWalker-v2
        # Pendulum-v0
        # self.env_name = 'MountainCarContinuous-v0'

        self.total_epochs = 10

        self.alpha = 0.1
        self.ac_kwargs = dict(hidden_sizes=[400, 300])
        self.gamma = 0.99
        self.replay_size = int(1e6)
        self.lr = 1e-3
        self.polyak = 0.995

        self.steps_per_epoch = 5000
        self.batch_size = 100
        self.start_steps = 10000
        self.max_ep_len = 1000
        self.save_freq = 1



        # TODO remove
        self.hid = 300
        self.l = 1
        # ---

        self.seed = 0

        self.max_exploration_episodes = 500
        self.batch_size = 256      # batch size during training
        self.rm_size = 1000000    # memory replay maximum size
        self.gamma = 0.99     # Discount factor
        self.critic_lr = 0.001    # Learning rate for critic
        self.actor_lr = 0.0001   # Learning rate for actor

        self.tau = 0.001    # moving average for target network

        self.max_episodes = 100

        self.valid_freq = 100
        # self.train_steps = 5

        self.train = True
        self.continue_training = False

        exp_name = "dsac1_" + self.env_name + "_" + str(datetime.datetime.now())

        self.logger_kwargs = setup_logger_kwargs(exp_name, self.seed)

        self.summary_dir = './tboard_ddpg'  # Directory for storing tensorboard summary results
        self.save_dir = './model_ddpg'      # Directory for storing trained model

        self.parameter_servers = ["localhost:2222"]
        self.workers = []
        for i in range(workers_num):
            self.workers.append("localhost:"+str(2223+i))
        self.workers_num = workers_num
