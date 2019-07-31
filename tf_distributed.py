import argparse
import sys

import tensorflow as tf
import time

from DDDPG.replay_buffer import ReplayBuffer

FLAGS = None


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
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            varstep = tf.Variable(5.)
            varstep_op = tf.assign_add(varstep, 1)
            step_op = global_step.assign(global_step + 1)
            init_op = tf.global_variables_initializer()

            replay_buffer = ReplayBuffer(100, 0)

        with tf.Session(server.target) as sess:
            sess.run(init_op)
            for _ in range(10):
                s, a, r, t, s2 = 1, 2, 3, 4, 5
                time.sleep(1)
                if FLAGS.task_index == 0:
                    print(server.target)
                    replay_buffer.add(s, a, r, t, s2)
                    sess.run(step_op)
                    sess.run(varstep_op)
                else:
                    # replay_buffer.add(s, a, r, t, s2)
                    print(replay_buffer.count)
                    # print(sess.run(global_step))
                    print(sess.run(varstep))

        # # The StopAtStepHook handles stopping after running given steps.
        # hooks = [tf.train.StopAtStepHook(last_step=1000000)]
        #
        # # The MonitoredTrainingSession takes care of session initialization,
        # # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # # or an error occurs.
        # with tf.train.MonitoredTrainingSession(master=server.target,
        #                                        is_chief=(FLAGS.task_index == 0),
        #                                        checkpoint_dir="/tmp/train_logs",
        #                                        hooks=hooks) as mon_sess:
        #     while not mon_sess.should_stop():
        #         # Run a training step asynchronously.
        #         # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        #         # perform *synchronous* training.
        #         # mon_sess.run handles AbortedError in case of preempted PS.
        #         mon_sess.run(step_op)


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
