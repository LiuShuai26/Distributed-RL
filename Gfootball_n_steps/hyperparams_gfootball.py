import numpy as np
import os
import sys
from gym.spaces import Box
import datetime
import gfootball.env as football_env
from numbers import Number


class HyperParameters:
    def __init__(self, env_name, exp_name, total_epochs, num_workers, a_l_ratio, weights_file):
        # parameters set

        self.env_name = "academy_3_vs_1_with_keeper"

        # "_random", "_d_True", ""
        self.rollout_env_name = self.env_name + ""
        self.exp_name = str(exp_name)

        self.with_checkpoints = False

        self.representation = 'simple115'
        self.stacked = False
        self.model = "mlp"

        self.Ln = 1

        self.a_l_ratio = a_l_ratio
        self.weights_file = weights_file

        # gpu memory fraction
        self.gpu_fraction = 0.2

        self.ac_kwargs = dict(hidden_sizes=[600, 400, 200])

        env_football = football_env.create_environment(env_name=self.env_name, stacked=self.stacked,
                                                       representation=self.representation, render=False)

        # env = FootballWrapper(env_football)
        env = env_football

        # self.obs_dim = env.observation_space.shape
        #
        # # self.obs_dim = 51
        # self.obs_space = env.observation_space
        # # self.obs_space = Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        # # TODO gfootball 1.3
        # if len(self.obs_dim) == 1 and self.env_name == "academy_3_vs_1_with_keeper":
        #     # self.obs_dim = (self.obs_dim,)
        #     self.obs_dim = (51, )
        #     self.obs_space = Box(low=-1.0, high=1.0, shape=(51,), dtype=np.float32)
        #
        # self.act_dim = env.action_space.n
        # self.act_space = env.action_space

        scenario_obsdim = {'academy_empty_goal': 32, 'academy_empty_goal_random': 32, 'academy_3_vs_1_with_keeper': 51,
                           'academy_3_vs_1_with_keeper_random': 51, 'academy_single_goal_versus_lazy': 108}
        self.obs_dim = scenario_obsdim[self.env_name]
        self.obs_space = Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

        self.o_shape = self.obs_space.shape

        self.act_dim = env.action_space.n
        self.act_space = env.action_space

        self.a_shape = self.act_space.shape

        # Share information about action space with policy architecture
        self.ac_kwargs['action_space'] = env.action_space

        self.total_epochs = total_epochs
        self.num_workers = num_workers


        self.Ln = 3
        self.use_max = False
        self.alpha = 0.1
        # self.alpha = "auto"
        self.target_entropy = 0.5

        self.gamma = 0.997
        self.replay_size = int(3e6)

        self.lr = 5e-5
        self.polyak = 0.995

        self.steps_per_epoch = 5000
        self.batch_size = 300
        self.start_steps = int(3e4)
        self.start_steps_per_worker = int(self.start_steps/self.num_workers)
        self.max_ep_len = 500
        self.save_freq = 1

        self.max_ret = 0

        self.seed = 0

        cwd = os.getcwd()

        self.summary_dir = cwd + '/tboard_ray'  # Directory for storing tensorboard summary results
        self.save_dir = cwd + '/' + self.exp_name    # Directory for storing trained model
        self.is_restore = False


# reward wrapper
class FootballWrapper(object):

    def __init__(self, env):
        self._env = env
        self.dis_to_goal = 0.0

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self):
        obs = self._env.reset()
        self.dis_to_goal = np.linalg.norm(obs[0:2] - [1.0, 0.0])
        return obs

    def step(self, action):
        r = 0.0
        for _ in range(1):
            obs, reward, done, info = self._env.step(action)
            # if reward != 0.0:
            #     done = True
            # else:
            #     done = False
            if reward < 0.0:
                reward = 0.0
            # reward -= 0.00175

            # if obs[0] < 0.0:
            #     done = True

            if not done:  # when env is done, ball position will be reset.
                reward += self.incentive(obs)

            r += reward

            if done:
                return obs, r * 150, done, info

        return obs, r * 150, done, info

    def incentive(self, obs):
        # total accumulative incentive reward is around 0.5
        dis_to_goal_new = np.linalg.norm(obs[0:2] - [1.01, 0.0])
        r = 0.25 * (self.dis_to_goal - dis_to_goal_new)
        self.dis_to_goal = dis_to_goal_new
        return r

    def incentive1(self, obs):
        r = -self.dis_to_goal * (1e-4)  # punishment weighted by dis_to_goal
        self.dis_to_goal = np.linalg.norm(obs[0:2] - [1.01, 0.0])  # interval: 0.0 ~ 2.0
        return r