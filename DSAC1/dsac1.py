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

from numbers import Number
import gym
import time
import core
from core import get_vars
from spinup.utils.logx import EpochLogger


from sklearn.preprocessing import StandardScaler

from replay_buffer import ReplayBuffer

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("workers_num", 1, "number of workers")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS


#   Agent Training
def train(sess, env, replay_buffer, x_ph, test_env, max_ep_len, logger, steps_per_epoch, epochs, start_steps,
          batch_size, x2_ph, a_ph, r_ph, d_ph, step_ops, save_freq, is_chief):
    def get_action(o, deterministic=False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1,-1)})[0]

    def test_agent(n=10):
        global sess, mu, pi, q1, q2, q1_pi, q2_pi
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy. 
        """
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of episode. Training (ep_len times).
        if d or (ep_len == max_ep_len):
            """
            Perform all SAC updates at the end of the trajectory.
            This is a slight difference from the SAC specified in the
            original paper.
            """
            for j in range(ep_len):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done'],
                            }
                # step_ops = [pi_loss, q1_loss, q2_loss, q1, q2, logp_pi, alpha, train_pi_op, train_value_op, target_update]
                outs = sess.run(step_ops, feed_dict)
                logger.store(LossPi=outs[0], LossQ1=outs[1], LossQ2=outs[2],
                            Q1Vals=outs[3], Q2Vals=outs[4],
                            LogPi=outs[5], Alpha=outs[6])

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0


        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0 and is_chief:
            epoch = t // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # logger.store(): store the data; logger.log_tabular(): log the data; logger.dump_tabular(): write the data
            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
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


def save_model(sess, saver, opt, global_step):
    save_path = saver.save(sess, opt.save_dir + "/model-workers_num:"+str(opt.workers_num), global_step=global_step)
    print('-------------------------------------')
    print("Model saved in file: %s" % save_path)
    print('-------------------------------------')


def main(_):

    opt = Parameters(FLAGS.workers_num)
    np.random.seed(opt.seed)
    tf.set_random_seed(opt.seed)


    ac_kwargs=dict(hidden_sizes=[400, 300])
    gamma=0.99
    epochs=500
    alpha=0.1
    replay_size = int(1e6)
    lr = 1e-3
    polyak = 0.995

    steps_per_epoch = 5000
    batch_size = 100
    start_steps = 10000,
    max_ep_len = 1000
    save_freq = 1

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(opt.env_name, opt.seed)
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    if opt.train:
        cluster = tf.train.ClusterSpec({"ps":opt.parameter_servers, "worker":opt.workers})
        server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

        if FLAGS.job_name == "ps":
            server.join()
        elif FLAGS.job_name == "worker":
            with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,cluster=cluster)):
                is_chief = (FLAGS.task_index == 0)
                # count the number of updates
                # global_step = tf.get_variable('global_step',[],initializer = tf.constant_initializer(0),trainable = False)
                # step_op = global_step.assign(global_step+1)

                env = gym.make(opt.env_name)

                actor_critic = core.mlp_actor_critic

                # ^--------------------------------------------------------------------------------------------



                env, test_env = gym.make(opt.env_name), gym.make(opt.env_name)
                obs_dim = env.observation_space.shape[0]
                act_dim = env.action_space.shape[0]

                # Action limit for clamping: critically, assumes all dimensions share the same bound!
                act_limit = env.action_space.high[0]

                # Share information about action space with policy architecture
                ac_kwargs['action_space'] = env.action_space

                # Inputs to computation graph
                x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

                # Main outputs from computation graph
                with tf.variable_scope('main'):
                    mu, pi, logp_pi, logp_pi2, q1, q2, q1_pi, q2_pi = actor_critic(x_ph, x2_ph, a_ph, **ac_kwargs)

                # Target value network
                with tf.variable_scope('target'):
                    _, _, logp_pi_, _, _, _, q1_pi_, q2_pi_ = actor_critic(x2_ph, x2_ph, a_ph, **ac_kwargs)

                # Experience buffer
                replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

                # Count variables
                var_counts = tuple(core.count_vars(scope) for scope in
                                   ['main/pi', 'main/q1', 'main/q2', 'main'])
                print(('\nNumber of parameters: \t pi: %d, \t' + \
                       'q1: %d, \t q2: %d, \t total: %d\n') % var_counts)

                ######
                if alpha == 'auto':
                    target_entropy = (-np.prod(env.action_space.shape))

                    log_alpha = tf.get_variable('log_alpha', dtype=tf.float32, initializer=0.0)
                    alpha = tf.exp(log_alpha)

                    alpha_loss = tf.reduce_mean(-log_alpha * tf.stop_gradient(logp_pi + target_entropy))

                    alpha_optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='alpha_optimizer')
                    train_alpha_op = alpha_optimizer.minimize(loss=alpha_loss, var_list=[log_alpha])
                ######

                # Min Double-Q:
                min_q_pi = tf.minimum(q1_pi_, q2_pi_)

                # Targets for Q and V regression
                v_backup = tf.stop_gradient(min_q_pi - alpha * logp_pi2)
                q_backup = r_ph + gamma * (1 - d_ph) * v_backup

                # Soft actor-critic losses
                pi_loss = tf.reduce_mean(alpha * logp_pi - q1_pi)
                q1_loss = 0.5 * tf.reduce_mean((q_backup - q1) ** 2)
                q2_loss = 0.5 * tf.reduce_mean((q_backup - q2) ** 2)
                value_loss = q1_loss + q2_loss

                # Policy train op
                # (has to be separate from value train op, because q1_pi appears in pi_loss)
                pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
                train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

                # Value train op
                # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
                value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
                value_params = get_vars('main/q')
                with tf.control_dependencies([train_pi_op]):
                    train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

                # Polyak averaging for target variables
                # (control flow because sess.run otherwise evaluates in nondeterministic order)
                with tf.control_dependencies([train_value_op]):
                    target_update = tf.group([tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
                                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

                # All ops to call during one training step
                if isinstance(alpha, Number):
                    step_ops = [pi_loss, q1_loss, q2_loss, q1, q2, logp_pi, tf.identity(alpha),
                                train_pi_op, train_value_op, target_update]
                else:
                    step_ops = [pi_loss, q1_loss, q2_loss, q1, q2, logp_pi, alpha,
                                train_pi_op, train_value_op, target_update, train_alpha_op]

                # Initializing targets to match main variables
                target_init = tf.group([tf.assign(v_targ, v_main)
                                        for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])


                # ^--------------------------------------------------------------------------------------------

                # Add ops to save and restore all the variables.
                saver = tf.train.Saver(max_to_keep=5)

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
                    sess = tf.Session()
                    sess.run(tf.global_variables_initializer())
                    sess.run(target_init)

                    # Setup model saving
                    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph},
                                          outputs={'mu': mu, 'pi': pi, 'q1': q1, 'q2': q2})

                    # if is_chief:
                    #     writer = tf.summary.FileWriter(opt.summary_dir+"/" + str(datetime.datetime.now()) + "-" + opt.env_name + "-workers_num:"+str(opt.workers_num), sess.graph)
                    # else:
                    #     writer = None

                    for _ in range(opt.max_episodes):
                        '''
                        if sv.should_stop():
                            break
                        '''

                        # current_step = sess.run(global_step)
                        # if current_step > opt.max_episodes:
                        #     break

                        # Train normally
                        train(sess, env, replay_buffer, x_ph, test_env, max_ep_len, logger, steps_per_epoch, epochs,
                              start_steps, batch_size, x2_ph, a_ph, r_ph, d_ph, step_ops, save_freq, is_chief)

                        # if current_step > 30 and reward < 0:
                        #     break

                        # if current_step % opt.valid_freq == opt.valid_freq-1:
                        #     # test_r = test(sess, current_step, opt, env, actor, critic, valid_ops, valid_vars, writer)
                        #     save_model(sess, saver, opt, global_step)

                        # Increase global_step
                        # sess.run(step_op)

                #sv.stop()
                print('Done')
                killall = 'pkill -9 -f dsac1.py'
                os.system(killall)


    else:       # For testing
        pass


if __name__ == '__main__':
    tf.app.run()
