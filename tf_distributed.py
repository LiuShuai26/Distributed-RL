import argparse
import sys

import tensorflow as tf
import numpy as np
import time

from DDDPG.replay_buffer import ReplayBuffer

FLAGS = None

import ray
from ray.utils import (decode, binary_to_object_id, binary_to_hex,
                       hex_to_binary)


@ray.remote
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def count(self):
        return self.size

# if FLAGS.job_name == "ps":
#     buffer = ReplayBuffer.remote(3, 3, 10)
#     buffer_id = ray.put(buffer)


def main(_):
    ps_hosts = ["localhost:2222"]
    worker_hosts = ["localhost:2223", "localhost:2224"]

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,
                                                      cluster=cluster)):
            # global_step = tf.contrib.framework.get_or_create_global_step()
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(5), trainable=False)
            tfid = tf.get_variable('tfid', [], dtype=tf.string)

            step_op = global_step.assign(global_step + 1)

            init_op = tf.global_variables_initializer()

            ray.init(redis_address="192.168.100.35:6379")

        with tf.Session(server.target) as sess:

            sess.run(init_op)

            if FLAGS.task_index == 0:
                buffer = ReplayBuffer.remote(3, 3, 100)
                buffer_id = ray.put(buffer)
                buffer_str = str(buffer_id)
                id_op = tfid.assign(buffer_str[9:-1])
                sess.run(id_op)
            else:
                time.sleep(1)
                buffer_oid = sess.run(tfid)
                print(buffer_oid)
                object_id = ray.ObjectID(hex_to_binary(buffer_oid))
                buffer = ray.get(object_id)
                print(buffer)
            for _ in range(10):
                time.sleep(1)
                if FLAGS.task_index == 0:
                    obs, act, rew, next_obs, done = np.ones((3)), np.ones((3)), 3, np.ones((3)), 1
                    buffer.store.remote(obs, act, rew, next_obs, done)
                    sess.run(step_op)
                    # print(ray.get(buffer.count.remote()))
                else:
                    print(sess.run(global_step))
                    # print(ray.get(buffer.count.remote()))
                    obs, act, rew, next_obs, done = np.ones((3)), np.ones((3)), 3, np.ones((3)), 1
                    buffer.store.remote(obs, act, rew, next_obs, done)
                    if ray.get(buffer.count.remote()) > 5:
                        print(ray.get(buffer.sample_batch.remote(5)))
                    print(ray.get(buffer.count.remote()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
