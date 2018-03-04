from baselines_utils import ortho_init
import constants as c
import glob
import numpy as np
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



class A2C:
    def __init__(self, sess, args, num_actions):
        self.sess = sess
        self.args = args
        self.num_actions = num_actions
        self.define_graph()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

        check_point = tf.train.get_checkpoint_state(self.args.model_load_dir)
        if check_point and check_point.model_checkpoint_path:
            print 'Restoring model from ' + check_point.model_checkpoint_path
            self.saver.restore(self.sess, check_point.model_checkpoint_path)

        #set up new writer
        self.summary_writer = tf.summary.FileWriter(self.args.summary_save_dir, self.sess.graph)


    def define_graph(self):
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.s_imgs_batch = tf.placeholder(dtype=tf.float32,
                                           shape=[None, c.PROTEIN_LENGTH,
                                                  c.PROTEIN_LENGTH, c.IN_CHANNELS])
        self.s_vector_batch = tf.placeholder(dtype=tf.int32,
                                             shape=[None, 2 * c.PROTEIN_LENGTH - 1])
        self.actions_taken = tf.placeholder(dtype=tf.int32)
        self.actor_labels = tf.placeholder(dtype=tf.float32)
        self.critic_labels = tf.placeholder(dtype=tf.float32)

        with tf.variable_scope("model", reuse=False):
            with tf.variable_scope("conv", reuse=False):
                #convs
                channel_sizes = [c.IN_CHANNELS] + c.CHANNEL_SIZES
                prev_layer = tf.cast(self.x_batch, tf.float32)
                for i in xrange(c.NUM_CONV_LAYERS):
                    stride = (1,) + c.CONV_STRIDES[i] + (1,)
                    conv_layer = tf.layers.conv2d(prev_layer, filters=channel_sizes[i + 1],
                                                  kernel_size=c.CONV_KERNEL_SIZES[i],
                                                  strides=stride, padding="VALID",
                                                  activation=tf.nn.relu, name="conv_%d" % i)
                    conv_layer = tf.nn.relu(conv_layer)
                    prev_layer = conv_layer

            #fully connected layers
            with tf.variable_scope("fc", reuse=False):
                conv_shape = cur_layer.shape
                flat_sz = conv_shape[1].value * conv_shape[2].value * conv_shape[3].value
                flattened = tf.reshape(cur_layer, shape=[-1, flat_sz])

                fc_layer = tf.layers.dense(flattened, c.FC_SIZE, name='fc_layer')
                fc_layer = tf.nn.relu(fc_layer)

            #policy output layer
            self.policy_logits = tf.layers.dense(fc_layer, self.num_actions, name='policy_logits')
            self.policy_probs = tf.nn.softmax(self.policy_logits)

            #value output layer
            self.value_preds = tf.layers.dense(fc_layer, 1, name='value_fc_layer')
            self.value_preds = tf.squeeze(self.value_preds, axis=1)

            params = tf.trainable_variables() #"model" scope's variables


        #intentionally defined outside of scope. Loss calculations:

        logpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.policy_logits, labels=self.actions_taken)

        self.actor_loss = tf.reduce_mean(self.actor_labels * logpac)
        self.critic_loss = tf.reduce_mean(tf.square(self.value_preds - self.critic_labels)) / 2.0

        self.entropy_regularization = tf.reduce_mean(self.calculate_entropy(self.policy_logits))
        self.actor_loss = self.actor_loss - c.ENTROPY_REGULARIZATION_WEIGHT * self.entropy_regularization

        self.total_loss = self.actor_loss + c.CRITIC_LOSS_WEIGHT * self.critic_loss

        self.optim = optim = tf.train.AdamOptimizer(learning_rate=c.LRATE)
        self.train_op = self.optim.minimize(self.total_loss, global_step=self.global_step)

        #summaries
        self.a_loss_summary = tf.summary.scalar("actor_loss", self.actor_loss)
        self.c_loss_summary = tf.summary.scalar("critic_loss", self.critic_loss)

        self.ep_reward = tf.placeholder(tf.float32)
        self.ep_reward_summary = tf.summary.scalar("episode_reward", self.ep_reward)

    def calculate_entropy(self, logits):
        a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=1)

    def get_values(self, s_batch):
        v_s = self.sess.run(self.value_preds, feed_dict={self.x_batch: s_batch})
        return v_s

    def train_step(self, s_batch, a_batch, r_batch):
        v_s = self.get_values(s_batch)
        advantage = r_batch - v_s #estimated k-step return - v_s
        k_step_return = np.reshape(r_batch, [-1]) #turn into row vec
        advantage = np.reshape(advantage, [-1]) #turn into row vec

        sess_args = [self.global_step, self.a_loss_summary, self.c_loss_summary, self.train_op]
        feed_dict = {self.x_batch: s_batch,
                    self.actor_labels: advantage,
                    self.critic_labels: k_step_return,
                    self.actions_taken: a_batch}
        step, a_summary, c_summary, _ = self.sess.run(sess_args, feed_dict=feed_dict)

        if (step - 1) % self.args.summary_save_freq == 0:
            self.summary_writer.add_summary(a_summary, global_step=step)
            self.summary_writer.add_summary(c_summary, global_step=step)

        if (step - 1) % self.args.model_save_freq == 0:
            self.saver.save(self.sess, os.path.join(self.args.model_save_dir, 'model'), global_step=step)

        return step

    def get_actions(self, states, action_validity):
        """
        Predict all Q values for a state -> softmax dist -> sample from dist

        :param states: A batch of states from the environment.
        :param valid_states: A vector of the valid actions we can take next.

        :return: A list of the action for each state
        """
        feed_dict = {self.x_batch: states}
        policy_probs = self.sess.run(self.policy_probs, feed_dict=feed_dict)
        # for each batch, look at the probabilities of just the valid states (passed in as indices)
        valid_actions = np.nonzero(action_validity)
        valid_policy_probs = policy_probs[valid_actions]
        actions = np.array([np.random.choice(len(state_probs), p=state_probs) for state_probs in valid_policy_probs])
        # get actual indices of actions we want to take, by accessing valid_states
        actions = valid_actions[actions]
        return actions


    def write_ep_reward_summary(self, ep_reward, steps):
        summary = self.sess.run(self.ep_reward_summary,
                                feed_dict={self.ep_reward: ep_reward})

        self.summary_writer.add_summary(summary, global_step=steps)

from utils import parse_args

if __name__ == '__main__':
    num_actions = 5
    sess = tf.Session()
    args = parse_args()
    model = A2C(sess, args, num_actions)
    batch_size = 10
    for _ in xrange(100):
        model.train_step(np.random.rand(batch_size,c.IN_WIDTH,c.IN_HEIGHT,c.IN_CHANNELS),
                        np.random.randint(num_actions, size=batch_size),
                        np.random.rand(batch_size),
                        1)
        print(model.get_actions(np.random.rand(1,84,84,4)))
