import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Concatenate, Input

weights_initializer = tf.initializers.GlorotUniform()

class Critic:
    def __init__(self, state_size, learning_rate, name='critic'):
        self.state_size = state_size
        self.learning_rate = learning_rate

        inputs = Input(self.state_size)
        # 5-Hidden Layers
        X = Dense(256, input_shape=(self.state_size,), activation="relu", kernel_initializer=weights_initializer,
                  name='h1')(inputs)
        X = Dense(160, activation="relu", kernel_initializer=weights_initializer, name='h2')(X)
        X = Dense(128, activation="relu", kernel_initializer=weights_initializer, name='h3')(X)
        X = Dense(64, activation="relu", kernel_initializer=weights_initializer, name='h4')(X)
        X = Dense(128, activation="relu", kernel_initializer=weights_initializer, name='h5')(X)
        # Output layer
        output = Dense(1, activation=None, kernel_initializer=weights_initializer, name='output')(X)

        self.model = Model(inputs=inputs, outputs=output, name='critic_model')

        # with tf.variable_scope(name):
        #     self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
        #     self.R_t = tf.placeholder(tf.float32, name="total_rewards")
        #     self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        #
        #     number_of_layers = 4
        #     weights = [256, 160, 128, 64, 128]
        #
        #     h = tf.layers.dense(units=weights[0], inputs=self.state, kernel_initializer=weights_initializer,
        #                         activation=tf.nn.relu)
        #
        #     for idx in range(1, number_of_layers):
        #         h = tf.layers.dense(units=weights[idx], inputs=h, kernel_initializer=weights_initializer,
        #                             activation=tf.nn.relu)
        #
        #     self.output = tf.layers.dense(units=1, inputs=h, kernel_initializer=weights_initializer,
        #                                   activation=None)
        #
        #     self.square_loss = tf.squared_difference(tf.squeeze(self.output), self.R_t)
        #     self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.square_loss)

    def predict(self, state):
        return self.model(state)

    def train(self, state, target, lr):

        with tf.GradientTape() as tape:
            optimizer = tf.optimizers.Adam(learning_rate=lr)
            output = self.predict(state)
            squeeze = tf.squeeze(output)
            loss = tf.math.squared_difference(squeeze, target)
            grads = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
