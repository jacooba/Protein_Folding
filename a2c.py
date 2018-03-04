import glob, os, utils
import constants as c, numpy as np, tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



class A2C:
    def __init__(self, sess, args, num_actions):
        self.sess = sess
        self.args = args
        self.num_actions = num_actions
        self.define_graph()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

        check_point = tf.train.get_checkpoint_state(c.MODEL_DIR)
        if check_point and check_point.model_checkpoint_path:
            print 'Restoring model from ' + check_point.model_checkpoint_path
            self.saver.restore(self.sess, check_point.model_checkpoint_path)

        #set up new writer
        self.summary_writer = tf.summary.FileWriter(c.SUMMARY_DIR, self.sess.graph)


    def define_placeholders():
        self.s_imgs_batch = tf.placeholder(dtype=tf.float32,
                                           shape=[None, c.IN_WIDTH,
                                                  c.IN_HEIGHT, c.IN_CHANNELS])
        self.s_vector_batch = tf.placeholder(dtype=tf.int32,
                                             shape=[None, 2 * c.L - 1])
        self.actions_taken = tf.placeholder(dtype=tf.int32)
        self.actor_labels = tf.placeholder(dtype=tf.float32)
        self.critic_labels = tf.placeholder(dtype=tf.float32)


    def define_graph(self):
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.define_placeholders()

        with tf.variable_scope("model", reuse=False):
            flattened_small_conv = utils.conv_layers("small_conv",
                                                    self.s_imgs_batch,
                                                    c.SMALL_CONV_CHANNELS,
                                                    c.SMALL_CONV_KERNELS,
                                                    c.SMALL_CONV_STRIDES)

            flattened_big_conv = utils.conv_layers("big_conv",
                                                  self.s_imgs_batch,
                                                  c.BIG_CONV_CHANNELS,
                                                  c.BIG_CONV_KERNELS,
                                                  c.BIG_CONV_STRIDES)

            flattened = tf.concat([flattened_small_conv, flattened_big_conv],
                                   axis=1, name="flattened_conv")

            fc_output = utils.fc_layers("fc", flattened, c.FC_SIZES)

            #policy output layer
            self.policy_logits = tf.layers.dense(fc_output, self.num_actions, name='policy_logits')
            self.policy_probs = tf.nn.softmax(self.policy_logits)

            #value output layer
            self.value_preds = tf.layers.dense(fc_output, 1, name='value_fc_layer')
            self.value_preds = tf.squeeze(self.value_preds, axis=1)

        self.define_losses_and_summaries()

    def define_losses_and_summaries():
        logpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.policy_logits,
                                                                labels=self.actions_taken)

        self.entropy = tf.reduce_mean(utils.entropy(self.policy_probs))
        self.actor_loss = tf.reduce_mean(self.actor_labels * logpac)
        self.actor_loss = self.actor_loss - c.ENTROPY_REGULARIZATION_WEIGHT * self.entropy

        self.critic_loss = tf.losses.mean_squared_error(self.critic_labels, self.value_preds)

        self.total_loss = self.actor_loss + c.CRITIC_LOSS_WEIGHT * self.critic_loss

        self.optim = optim = tf.train.AdamOptimizer(learning_rate=c.LRATE)
        self.train_op = self.optim.minimize(self.total_loss, global_step=self.global_step)

        #summaries
        self.a_loss_summary = tf.summary.scalar("actor_loss", self.actor_loss)
        self.c_loss_summary = tf.summary.scalar("critic_loss", self.critic_loss)
        self.ep_reward = tf.placeholder(tf.float32)
        self.ep_reward_summary = tf.summary.scalar("episode_reward", self.ep_reward)


    def get_values(self, s_imgs_batch, s_vector_batch):
        return self.value_preds.eval(feed_dict={self.s_imgs_batch: s_imgs_batch,
                                                self.s_vector_batch: s_vector_batch})


    def train_step(self, batch_triple, a_batch, r_batch):
        s_imgs_batch, s_vector_batch = utils.unpack_batch_triple(batch_triple)

        v_s = self.get_values(s_imgs_batch, s_vector_batch)
        advantage = r_batch - v_s #estimated k-step return - v_s
        k_step_return = np.reshape(r_batch, [-1]) #turn into row vec
        advantage = np.reshape(advantage, [-1]) #turn into row vec

        sess_args = [self.global_step, self.a_loss_summary, self.c_loss_summary, self.train_op]
        feed_dict = {self.s_imgs_batch: s_imgs_batch,
                     self.s_vector_batch: s_vector_batch,
                    self.actor_labels: advantage,
                    self.critic_labels: k_step_return,
                    self.actions_taken: a_batch}
        step, a_summary, c_summary, _ = self.sess.run(sess_args, feed_dict=feed_dict)

        if (step - 1) % c.SUMMARY_SAVE_FREQ == 0:
            self.summary_writer.add_summary(a_summary, global_step=step)
            self.summary_writer.add_summary(c_summary, global_step=step)

        if (step - 1) % c.MODEL_SAVE_FREQ == 0:
            self.saver.save(self.sess, os.path.join(c.MODEL_DIR, 'model'), global_step=step)

        return step


    def get_actions(self, batch_triple, action_validity):
        """
        Predict all Q values for a state -> softmax dist -> sample from dist

        :param states: A batch of states from the environment.
        :param valid_states: A vector of the valid actions we can take next.

        :return: A list of the action for each state
        """
        s_imgs_batch, s_vector_batch = utils.unpack_batch_triple(batch_triple)

        feed_dict = {self.s_imgs_batch: s_imgs_batch, self.s_vector_batch: s_vector_batch}
        policy_probs = self.policy_probs.eval(feed_dict=feed_dict)
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
