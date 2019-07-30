import tensorflow as tf
import numpy as np
from parameters import Parameters
import gym
from gym import wrappers
import tflearn
import sys
import os
import time
import datetime
import itertools
from sklearn.preprocessing import StandardScaler

from networks import ActorNetwork, CriticNetwork

opt = Parameters()
env = gym.make(opt.env_name)
# env = wrappers.Monitor(env, './tmp/', force=True)


if opt.env_name == 'MountainCarContinuous-v0':
    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    scaler = StandardScaler()
    scaler.fit(observation_examples)
else:
    scaler = None

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_scale = abs(env.action_space.high[0])
actor = ActorNetwork(state_dim, action_dim, action_scale, opt.actor_lr, opt.tau, scaler)
critic = CriticNetwork(state_dim, action_dim, opt.critic_lr, opt.tau, actor.get_num_trainable_vars(), scaler)
saver = tf.train.Saver(max_to_keep=5)


num_tests = 10


with tf.Session() as sess:
    actor.set_session(sess)
    critic.set_session(sess)
    saver.restore(sess, tf.train.latest_checkpoint(opt.save_dir + '/'))
    actor.restore_params(tf.trainable_variables())
    critic.restore_params(tf.trainable_variables())

    total_reward = 0
    for _ in range(num_tests):
        # env.render()
        state = env.reset()

        reward = 0
        for t in itertools.count():
            input_s = np.reshape(state, (1, actor.s_dim))
            a = actor.predict_target(input_s)

            state2, r, done, _ = env.step(a[0])
            env.render()
            reward += r
            state = state2

            if done:
                print(reward)
                total_reward += reward
                break
    print("mean reward:", total_reward/num_tests)
