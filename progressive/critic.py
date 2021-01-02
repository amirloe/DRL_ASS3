import tensorflow
import tensorflow.compat.v1 as tf

weights_initializer = tensorflow.initializers.GlorotUniform()

class ProgCritic:
    def __init__(self, critic1,  critic2,  state_size, learning_rate, name='critic'):
        self.state_size = state_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

            self.W1 = tf.get_variable("W1", [self.state_size, 12],
                                      initializer=tensorflow.initializers.variance_scaling(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [12, 1], initializer=tensorflow.initializers.variance_scaling(seed=0))
            self.b2 = tf.get_variable("b2", [1], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)

            # self.output_in = tf.add(self.A1, tf.add(critic1.A1, critic2.A1))
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            self.test_out = tf.add(self.output, tf.add(critic1.output,critic2.output))



            self.square_loss = tf.squared_difference(tf.squeeze(self.test_out), self.R_t)

            tvars = tf.trainable_variables()
            trainable_vars = [var for var in tvars if name in var.name]

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.square_loss, var_list=trainable_vars)