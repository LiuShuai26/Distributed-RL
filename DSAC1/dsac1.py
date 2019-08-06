import tensorflow as tf
import numpy as np
from parameters import Parameters
import os
from numbers import Number
import gym
import time
import core
from core import get_vars
from core import mlp_actor_critic as actor_critic
from spinup.utils.logx import EpochLogger
from replay_buffer import ReplayBuffer

import ray
from ray.utils import hex_to_binary


flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_string("env_name", "", "game env")
flags.DEFINE_integer("total_epochs", 100, "total_epochs")
flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
flags.DEFINE_integer("workers_num", 1, "number of workers")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")


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


#   Agent Training
def train(sess, env, replay_buffer, x_ph, test_env, logger, x2_ph, a_ph, r_ph, d_ph, step_ops, is_chief,
          current_epoch, mu, pi, start_time, opt):

    def get_action(o, deterministic=False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1, -1)})[0]

    def test_agent(n=10):
        global sess, mu, pi, q1, q2, q1_pi, q2_pi
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == opt.max_ep_len)):
                # Take deterministic actions at test time
                # test_env.render()
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_reward = 0
    episodes = 0
    # Main loop: collect experience in env and update/log each epoch
    for _ in range(opt.steps_per_epoch):
        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy. 
        """

        # get action by policy after first "two" epochs
        if current_epoch > 1:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        # env.render()
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        # TODO remove
        d = False if ep_len == opt.max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store.remote(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of episode. Training (ep_len times).
        if d or (ep_len == opt.max_ep_len):
            """
            Perform all SAC updates at the end of the trajectory.
            This is a slight difference from the SAC specified in the
            original paper.
            """
            for j in range(ep_len):
                batch = ray.get(replay_buffer.sample_batch.remote(opt.batch_size))
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done'],
                             }

                outs = sess.run(step_ops, feed_dict)
                if is_chief:
                    logger.store(LossPi=outs[0], LossQ1=outs[1], LossQ2=outs[2],
                                Q1Vals=outs[3], Q2Vals=outs[4],
                                LogPi=outs[5], Alpha=outs[6])
            total_reward += ep_ret
            episodes += 1
            if is_chief:
                logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    print("epoch:", current_epoch, "average reward:", total_reward/episodes)

    # End of epoch wrap-up
    if is_chief:

        # Save model
        if (current_epoch % opt.save_freq == 0) or (current_epoch == opt.total_epochs-1):
            logger.save_state({'env': env}, None)

        # Test the performance of the deterministic version of the agent.
        test_agent()

        # logger.store(): store the data; logger.log_tabular(): log the data; logger.dump_tabular(): write the data
        # Log info about epoch
        logger.log_tabular('Epoch', current_epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('TestEpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('TestEpLen', average_only=True)
        # TODO line below might not good
        logger.log_tabular('TotalEnvInteracts', opt.steps_per_epoch*(current_epoch+1))
        logger.log_tabular('Alpha',average_only=True)
        logger.log_tabular('Q1Vals', with_min_and_max=True)
        logger.log_tabular('Q2Vals', with_min_and_max=True)
        # logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('LogPi', with_min_and_max=True)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossQ1', average_only=True)
        logger.log_tabular('LossQ2', average_only=True)
        # logger.log_tabular('LossV', average_only=True)

        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()


def main(_):

    opt = Parameters(FLAGS.env_name, FLAGS.total_epochs, FLAGS.workers_num)

    np.random.seed(opt.seed)
    tf.set_random_seed(opt.seed)

    if opt.train:
        cluster = tf.train.ClusterSpec({"ps": opt.parameter_servers, "worker": opt.workers})
        server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

        if FLAGS.job_name == "ps":
            server.join()
        elif FLAGS.job_name == "worker":
            # Variable is placed in the parameter server by the replica_device_setter
            with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,
                                                          cluster=cluster)):

                is_chief = (FLAGS.task_index == 0)
                if is_chief:
                    logger = EpochLogger(**opt.logger_kwargs)
                    logger.save_config(locals())
                else:
                    logger = None

                # count the number of updates
                global_epoch = tf.get_variable('global_epoch', [], initializer=tf.constant_initializer(0),
                                               trainable=False)
                epoch_op = global_epoch.assign(global_epoch+1)

                # ray
                # TODO redis_address
                ray.init(redis_address="192.168.100.126:6379")
                buffer_id_str = tf.get_variable('buffer_id_str', [], dtype=tf.string)

                # --------------------------- sac1 graph part start --------------------------------
                env, test_env = gym.make(opt.env_name), gym.make(opt.env_name)
                obs_dim = env.observation_space.shape[0]
                act_dim = env.action_space.shape[0]

                # Action limit for clamping: critically, assumes all dimensions share the same bound!
                act_limit = env.action_space.high[0]

                # Share information about action space with policy architecture
                opt.ac_kwargs['action_space'] = env.action_space

                # Inputs to computation graph
                x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

                # Main outputs from computation graph
                with tf.variable_scope('main'):
                    mu, pi, logp_pi, logp_pi2, q1, q2, q1_pi, q2_pi = actor_critic(x_ph, x2_ph, a_ph, **opt.ac_kwargs)

                # Target value network
                with tf.variable_scope('target'):
                    _, _, logp_pi_, _, _, _, q1_pi_, q2_pi_ = actor_critic(x2_ph, x2_ph, a_ph, **opt.ac_kwargs)

                # Count variables
                var_counts = tuple(core.count_vars(scope) for scope in
                                   ['main/pi', 'main/q1', 'main/q2', 'main'])
                if is_chief:
                    print(('\nNumber of parameters: \t pi: %d, \t' + 'q1: %d, \t q2: %d, \t total: %d\n') % var_counts)

                # TODO alpha never use
                ######
                if opt.alpha == 'auto':
                    target_entropy = (-np.prod(env.action_space.shape))

                    log_alpha = tf.get_variable('log_alpha', dtype=tf.float32, initializer=0.0)
                    alpha = tf.exp(log_alpha)

                    alpha_loss = tf.reduce_mean(-log_alpha * tf.stop_gradient(logp_pi + target_entropy))

                    alpha_optimizer = tf.train.AdamOptimizer(learning_rate=opt.lr, name='alpha_optimizer')
                    train_alpha_op = alpha_optimizer.minimize(loss=alpha_loss, var_list=[log_alpha])
                ######

                # Min Double-Q:
                min_q_pi = tf.minimum(q1_pi_, q2_pi_)

                # Targets for Q and V regression
                v_backup = tf.stop_gradient(min_q_pi - opt.alpha * logp_pi2)
                q_backup = r_ph + opt.gamma * (1 - d_ph) * v_backup

                # Soft actor-critic losses
                pi_loss = tf.reduce_mean(opt.alpha * logp_pi - q1_pi)
                q1_loss = 0.5 * tf.reduce_mean((q_backup - q1) ** 2)
                q2_loss = 0.5 * tf.reduce_mean((q_backup - q2) ** 2)
                value_loss = q1_loss + q2_loss

                # Policy train op
                # (has to be separate from value train op, because q1_pi appears in pi_loss)
                pi_optimizer = tf.train.AdamOptimizer(learning_rate=opt.lr)
                train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

                # Value train op
                # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
                value_optimizer = tf.train.AdamOptimizer(learning_rate=opt.lr)
                value_params = get_vars('main/q')
                with tf.control_dependencies([train_pi_op]):
                    train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

                # Polyak averaging for target variables
                # (control flow because sess.run otherwise evaluates in nondeterministic order)
                with tf.control_dependencies([train_value_op]):
                    target_update = tf.group([tf.assign(v_targ, opt.polyak * v_targ + (1 - opt.polyak) * v_main)
                                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

                # All ops to call during one training step
                if isinstance(opt.alpha, Number):
                    step_ops = [pi_loss, q1_loss, q2_loss, q1, q2, logp_pi, tf.identity(opt.alpha),
                                train_pi_op, train_value_op, target_update]
                else:
                    step_ops = [pi_loss, q1_loss, q2_loss, q1, q2, logp_pi, opt.alpha,
                                train_pi_op, train_value_op, target_update, train_alpha_op]

                # Initializing targets to match main variables
                target_init = tf.group([tf.assign(v_targ, v_main)
                                        for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

                # --------------------------- sac1 graph part end --------------------------------

                # TODO should have restore model feature

                # Add ops to save and restore all the variables.
                # saver = tf.train.Saver(max_to_keep=5)

                # if opt.continue_training:
                #     def restore_model(sess):
                #         actor.set_session(sess)
                #         critic.set_session(sess)
                #         saver.restore(sess,tf.train.latest_checkpoint(opt.save_dir+'/'))
                #         actor.restore_params(tf.trainable_variables())
                #         critic.restore_params(tf.trainable_variables())
                #         print('***********************')
                #         print('Model Restored')
                #         print('***********************')
                # else:
                #     def restore_model(sess):
                #         actor.set_session(sess)
                #         critic.set_session(sess)
                #         # Initialize target network weights
                #         actor.update_target_network()
                #         critic.update_target_network()
                #         print('***********************')
                #         print('Model Initialized')
                #         print('***********************')

                with tf.Session(server.target) as sess:
                    # if i
                    sess.run(tf.global_variables_initializer())
                    sess.run(target_init)
                    if is_chief:
                        # Experience buffer
                        replay_buffer = ReplayBuffer.remote(obs_dim=obs_dim, act_dim=act_dim, size=opt.replay_size)
                        buffer_id = ray.put(replay_buffer)
                        buffer_id_op = buffer_id_str.assign(str(buffer_id)[9:-1])
                        sess.run(buffer_id_op)
                    else:
                        # wait chief put buffer in ray
                        time.sleep(1)
                        buffer_id = ray.ObjectID(hex_to_binary(sess.run(buffer_id_str)))
                        replay_buffer = ray.get(buffer_id)

                    if is_chief:
                        # Setup model saving
                        logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph},
                                              outputs={'mu': mu, 'pi': pi, 'q1': q1, 'q2': q2})

                    # if is_chief:
                    #     writer = tf.summary.FileWriter(opt.summary_dir+"/" + str(datetime.datetime.now()) + "-" +
                    #                                    opt.env_name + "-workers_num:" +
                    #                                    str(opt.workers_num), sess.graph)
                    # else:
                    #     writer = None

                    start_time = time.time()

                    epochs = opt.total_epochs // FLAGS.workers_num
                    # episode should be epoch
                    for _ in range(epochs):

                        current_epoch = sess.run(global_epoch)

                        # Train normally
                        train(sess, env, replay_buffer, x_ph, test_env, logger, x2_ph, a_ph, r_ph, d_ph, step_ops,
                              is_chief, current_epoch, mu, pi, start_time, opt)

                        # if current_step % opt.valid_freq == opt.valid_freq-1:
                        #     # test_r = test(sess, current_step, opt, env, actor, critic, valid_ops, valid_vars, writer)
                        #     save_model(sess, saver, opt, global_step)

                        # Increase global_step
                        sess.run(epoch_op)

                print('Done')

                # TODO might not work well
                if is_chief:
                    # kill_all = 'pkill -9 -f dsac1.py'
                    kill_all = './kill_all.sh'
                    os.system(kill_all)

    else:       # For testing
        pass


if __name__ == '__main__':
    tf.app.run()

