import tensorflow
import tensorflow.compat.v1 as tf

weights_initializer = tensorflow.initializers.GlorotUniform()


class ProgActor:
    def __init__(self, actor1,  actor2,  state_size, action_size, name='actor'):
        self.state_size = state_size
        self.action_size = action_size


        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")
            self.W1 = tf.get_variable("W1", [self.state_size, 32],
                                      initializer=tensorflow.initializers.variance_scaling(seed=0))
            self.b1 = tf.get_variable("b1", [32], initializer=tf.zeros_initializer())

            self.W2 = tf.get_variable("W2", [32, self.action_size],
                                      initializer=tensorflow.initializers.variance_scaling(seed=0))
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)

            self.A1 = tf.nn.relu(self.Z1)

            # # Prog addition
            # self.output_in = tf.add(self.A1, tf.add(actor1.A1, actor2.A1))

            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            self.test_out = tf.add(self.output, tf.add(actor1.output,actor2.output))

            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.test_out))
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)

            tvars = tf.trainable_variables()
            trainable_vars = [var for var in tvars if name in var.name]

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                               var_list=trainable_vars)
