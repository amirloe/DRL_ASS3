import numpy as np
import tensorflow as tf


def load_weights(sess, actor, critic, load_model_name):
    for model, model_name in zip([actor, critic], ['actor', 'critic']):
        weight_placeholder = tf.compat.v1.placeholder(tf.float32, tuple(model.W1.shape))
        assign_op = tf.compat.v1.assign(model.W1, weight_placeholder)
        sess.run(assign_op,
                 feed_dict={weight_placeholder: np.load(f'../trained_models/{load_model_name}_{model_name}_w1.npy')})

        weight_placeholder = tf.compat.v1.placeholder(tf.float32, tuple(model.b1.shape))
        assign_op = tf.compat.v1.assign(model.b1, weight_placeholder)
        sess.run(assign_op,
                 feed_dict={weight_placeholder: np.load(f'../trained_models/{load_model_name}_{model_name}_b1.npy')})

        weight_placeholder = tf.compat.v1.placeholder(tf.float32, tuple(model.W2.shape))
        assign_op = tf.compat.v1.assign(model.W2, weight_placeholder)
        sess.run(assign_op,
                 feed_dict={weight_placeholder: np.load(f'../trained_models/{load_model_name}_{model_name}_w2.npy')})

        weight_placeholder = tf.compat.v1.placeholder(tf.float32, tuple(model.b2.shape))
        assign_op = tf.compat.v1.assign(model.b2, weight_placeholder)
        sess.run(assign_op,
                 feed_dict={weight_placeholder: np.load(f'../trained_models/{load_model_name}_{model_name}_b2.npy')})