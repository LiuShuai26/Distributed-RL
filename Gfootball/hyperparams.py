import numpy as np
import os
import sys
import gym
import datetime
import gfootball.env as football_env


class FootballWrapper(object):

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        reward = reward + self.incentive(obs)
        return obs, reward, done, info

    def incentive(self, obs):
        who_controls_ball = obs[7:9]
        pos_ball = obs[0]
        distance_to_goal = np.array([(pos_ball + 1) / 2.0, (pos_ball - 1) / 2.0])
        r = np.dot(who_controls_ball, distance_to_goal) * 0.003
        return r


class HyperParameters:
    def __init__(self, total_epochs, num_workers=1, a_l_ratio=1):
        # parameters set

        # ray_servr_address = ""

        # self.env_name = 'LunarLanderContinuous-v2'   # 'MountainCarContinuous-v0'
        self.env_name = "gfootball"
        # BipedalWalker-v2
        # Pendulum-v0
        # self.env_name = 'MountainCarContinuous-v0'

        self.a_l_ratio = a_l_ratio

        # gpu memory fraction
        self.gpu_fraction = 0.3

        self.ac_kwargs = dict(hidden_sizes=[400, 300])

        env_football = football_env.create_environment(env_name='11_vs_11_easy_stochastic',
                                                       with_checkpoints=False, representation='simple115',
                                                       render=False)

        # env = FootballWrapper(env_football)
        env = env_football

        self.obs_dim = env.observation_space.shape[0]
        self.obs_space = env.observation_space
        self.act_dim = env.action_space.n
        self.act_space = env.action_space

        # Share information about action space with policy architecture
        self.ac_kwargs['action_space'] = env.action_space

        self.total_epochs = total_epochs
        self.num_workers = num_workers

        self.alpha = 0.1

        self.gamma = 0.997
        self.replay_size = 3000000
        # self.lr = 1e-4
        self.lr = 1e-3
        self.polyak = 0.995

        self.steps_per_epoch = 5000
        self.batch_size = 100
        self.start_steps = 10000
        self.max_ep_len = 1000
        self.save_freq = 1

        self.seed = 0

        self.summary_dir = './tboard_ray_sac1'  # Directory for storing tensorboard summary results
        self.save_dir = './model_ray_sac1'      # Directory for storing trained model
