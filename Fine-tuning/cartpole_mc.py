import os
import time
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
import tensorflow.compat.v1 as tf
import sklearn.preprocessing
import loader

tf.disable_v2_behavior()

env = gym.make('MountainCarContinuous-v0')
np.random.seed(12)
weights_initializer = tensorflow.initializers.GlorotUniform()

def pad_state(state_to_pad):
    state_to_pad = state_to_pad.reshape([1, 2])
    result = np.zeros((1, 6))
    result[0, :2] = state_to_pad
    return result

#np.append(env.observation_space.sample(), np.zeros(6 - env.observation_space.shape[0]))
state_space_samples = np.array(
    [np.append(env.observation_space.sample(), np.zeros(6 - env.observation_space.shape[0])) for x in range(10000)])
state_space_samples += np.append(env.observation_space.high, np.zeros(6 - env.observation_space.shape[0]))
state_space_samples += np.append(env.observation_space.low, np.zeros(6 - env.observation_space.shape[0]))
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(state_space_samples)


# function to normalize states
def scale_state(state_to_scale):  # requires input shape=(2,)
    scaled = scaler.transform([state_to_scale[0]])
    return scaled


class Actor:
    def __init__(self, state_size, action_size, name='actor'):
        self.state_size = state_size
        self.action_size = action_size

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")
            self.W1 = tf.get_variable("W1", [self.state_size, 32],
                                      initializer=tensorflow.initializers.variance_scaling(seed=123))
            self.b1 = tf.get_variable("b1", [32], initializer=tf.zeros_initializer())

            self.W2 = tf.get_variable("W2", [32, self.action_size],
                                      initializer=tensorflow.initializers.variance_scaling(seed=0))
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)

            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            tvars = tf.trainable_variables()
            trainable_vars = [var for var in tvars if '2' in var.name]


            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,var_list=trainable_vars)

    def predict(self, state):
        sess = tf.get_default_session()
        predict_feed_dict = {self.state: state}
        return sess.run(self.actions_distribution, predict_feed_dict)

    def train(self, state, td_error, action_one_hot, actor_lr):
        sess = tf.get_default_session()
        feed_dict_policy = {self.state: state,
                            self.R_t: td_error,
                            self.action: action_one_hot,
                            self.learning_rate: actor_lr}
        sess.run([self.optimizer, self.loss], feed_dict_policy)


class Critic:
    def __init__(self, state_size, learning_rate, name='critic'):
        self.state_size = state_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

            self.W1 = tf.get_variable("W1", [self.state_size, 12],
                                      initializer=tensorflow.initializers.variance_scaling(seed=123))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [12, 1], initializer=tensorflow.initializers.variance_scaling(seed=0))
            self.b2 = tf.get_variable("b2", [1], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            self.square_loss = tf.squared_difference(tf.squeeze(self.output), self.R_t)

            tvars = tf.trainable_variables()
            trainable_vars = [var for var in tvars if '2' in var.name]
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.square_loss,var_list=trainable_vars)

    def predict(self, state):
        sess = tf.get_default_session()
        predict_feed_dict = {self.state: state}
        return sess.run(self.output, predict_feed_dict)

    def train(self, state, target, lr):
        sess = tf.get_default_session()
        update_feed_dict = {self.state: state, self.R_t: target, self.learning_rate: lr}
        sess.run([self.optimizer, self.square_loss], update_feed_dict)


def train_actor_and_critic(actor, critic, state, next_state, done, reward, discount_factor, actor_lr, critic_lr,
                           action_one_hot):
    value = sess.run(critic.output, {critic.state: state})
    next_value = sess.run(critic.output, {critic.state: next_state})

    if done:
        td_target = reward
    else:
        td_target = reward + discount_factor * next_value

    td_error = td_target - value

    # Critic train
    update_feed_dict = {critic.state: state, critic.R_t: td_target, critic.learning_rate: critic_lr}
    sess.run([critic.optimizer, critic.square_loss], update_feed_dict)

    # Actor train
    feed_dict_policy = {actor.state: state,
                        actor.R_t: td_error,
                        actor.action: action_one_hot,
                        actor.learning_rate: actor_lr}
    sess.run([actor.optimizer, actor.loss], feed_dict_policy)


# Define hyperparameters
state_size = 6
action_size = 3

max_episodes = 5000
max_steps = 1000
discount_factor = 0.99
actor_lr = 0.0005
critic_lr = 0.01
learning_rate_decay = 1

EXPLOITING_PHASE_LENGTH = 7
epsilon = 1

render = False

# Initialize the AC networks
tf.reset_default_graph()
actor = Actor(state_size, action_size, "cartpole_actor")
critic = Critic(state_size, critic_lr, "cartpole_critic")

start_time = time.time()



with tf.Session() as sess:

    saver = tf.train.Saver()
    summary = tf.summary.FileWriter("../tensorboard/ft/cartpole_mc", sess.graph)
    # saver.restore(sess=sess, save_path='../data/cartpole/cartpole.h')
    sess.run(tf.global_variables_initializer())
    loader.load_weights(sess,actor,critic,'cartpole')
    solved = False

    # First, generate some data for initial rewards training
    episode_rewards = np.zeros(max_episodes)
    average_rewards = 0.0


    times_of_success_before_exploiting = 0
    for episode in range(max_episodes):
        state = env.reset()
        state = np.append(state, np.zeros(6 - env.observation_space.shape[0]))
        state = scale_state( state.reshape([1, state_size]))
        left_max, right_max = state[0][0], state[0][0]
        iter = 0
        done = False
        while not done and iter < max_steps:
            actions_distribution = sess.run(actor.actions_distribution, {actor.state: state})
            # if np.random.uniform(0, 1) < epsilon:
            #     action = np.random.choice(range(len(actions_distribution)))
            # else:
            action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
            while action >= 2:
                action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
            action_choices = [[-1.], [1.], [0.]]

            next_state, reward, done, _ = env.step(action_choices[action])
            # print(f'Reward = {reward}')
            # reward = -10 if reward == 0 else reward
            episode_rewards[episode] += reward
            next_state = np.append(next_state, np.zeros(6 - env.observation_space.shape[0]))
            next_state = scale_state(next_state.reshape([1, state_size]))

            if times_of_success_before_exploiting < EXPLOITING_PHASE_LENGTH:
                if reward <= 0:
                    if next_state[0][0] < left_max:
                        reward += 1.3
                        left_max = next_state[0][0]
                    if next_state[0][0] > right_max:
                        reward += 1.3
                        right_max = next_state[0][0]
                else:
                    times_of_success_before_exploiting += 1
                    if iter < 200:
                        reward += 30
                    else:
                        reward += 75
                    print(f"finished, reward: {reward}")

            if render and episode % 10 == 0:
                env.render()


            action_one_hot = np.zeros(action_size)
            action_one_hot[action] = 1
            train_actor_and_critic(actor, critic, state, next_state, done, reward, discount_factor,actor_lr,critic_lr,action_one_hot)
            if times_of_success_before_exploiting < EXPLOITING_PHASE_LENGTH:
                epsilon = max(epsilon * 0.995, 0.5)
            else:
                epsilon = max(epsilon * 0.995, 0.05)
            iter += 1
            if done:
                episode_summary = tf.Summary()
                episode_summary.value.add(tag="Iters", simple_value=episode_rewards[episode])
                summary.add_summary(episode_summary, episode)

                actor_lr = actor_lr * learning_rate_decay
                critic_lr = critic_lr * learning_rate_decay

                if episode > 50:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards[(episode - 49):episode + 1])
                print("Episode {} Reward: {} Average over 50 episodes: {}".format(episode, episode_rewards[episode],
                                                                                  round(average_rewards, 2)))
                print(f"took total of {iter} iters")
                if episode_rewards[episode] >= 85:
                    print('Solved at episode: ' + str(episode))
                    elapsed_time = time.time() - start_time
                    print(f"elapsed_time: {elapsed_time}")
                    solved = True
                break
            state = next_state
            max_episodes = max(max_episodes - 250, 1000)

        if solved:
            break
        saver.save(sess, save_path='../data/ft/mc/mc.h')

    plt.figure(figsize=(20, 10))
    non_zero_rewards = episode_rewards[:episode + 1]
    episodes = [i for i in range(len(non_zero_rewards))]
    plt.plot(episodes, non_zero_rewards, color='blue', marker='o')
    plt.title(f'Actor Critic algorithm. Episode till solve: {episode}', fontsize=14)
    plt.xlabel('episode', fontsize=14)
    plt.ylabel('Rewards', fontsize=14)
    plt.grid(True)
    plt.show()
