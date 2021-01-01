import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Concatenate, Input

weights_initializer = tf.initializers.GlorotUniform()


class Actor:
    def __init__(self, state_size, action_size, name='actor'):
        self.state_size = state_size
        self.action_size = action_size
        inputs = Input(self.state_size)
        # 5-Hidden Layers
        X = Dense(256, input_shape=(self.state_size,), activation="relu", kernel_initializer=weights_initializer,
                  name='h1')(inputs)
        X = Dense(160, activation="relu", kernel_initializer=weights_initializer, name='h2')(X)
        X = Dense(128, activation="relu", kernel_initializer=weights_initializer, name='h3')(X)
        X = Dense(64, activation="relu", kernel_initializer=weights_initializer, name='h4')(X)
        X = Dense(64, activation="relu", kernel_initializer=weights_initializer, name='h5')(X)
        # Output layer
        output = Dense(self.action_size, activation=None, kernel_initializer=weights_initializer, name='output')(X)

        self.model = Model(inputs=inputs, outputs=output, name='actor_model')

    def predict(self, state):
        return self.model(state)

    def train(self, state, td_error, action_one_hot, actor_lr):
        with tf.GradientTape() as tape:
            optimizer = tf.optimizers.Adam(learning_rate=actor_lr)
            output = self.predict(state)
            neg_log_prob = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=action_one_hot)
            loss = tf.reduce_mean(neg_log_prob * td_error)
            grads = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
