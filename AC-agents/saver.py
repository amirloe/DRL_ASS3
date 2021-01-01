import numpy as np
import tensorflow as tf


def save_weights(sess, actor, critic, save_model_name):
    for model, model_name in zip([actor, critic], ['actor', 'critic']):
        weight = sess.run(model.W1)
        np.save(f'../trained_models/{save_model_name}_{model_name}_w1.npy', weight)

        weight = sess.run(model.b1)
        np.save(f'../trained_models/{save_model_name}_{model_name}_b1.npy', weight)

        weight = sess.run(model.W2)
        np.save(f'../trained_models/{save_model_name}_{model_name}_w2.npy', weight)

        weight = sess.run(model.b2)
        np.save(f'../trained_models/{save_model_name}_{model_name}_b2.npy', weight)
