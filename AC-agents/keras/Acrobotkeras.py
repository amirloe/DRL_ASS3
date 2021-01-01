import time
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.actor import Actor
from keras.critic import Critic



env = gym.make('Acrobot-v1')
np.random.seed(1)
weights_initializer = tf.initializers.GlorotUniform()

#
# class Actor:
#     def __init__(self, state_size, action_size, name='actor'):
#         self.state_size = state_size
#         self.action_size = action_size
#
#         with tf.variable_scope(name):
#             self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
#             self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
#             self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
#             self.R_t = tf.placeholder(tf.float32, name="total_rewards")
#
#             number_of_layers = 4
#             weights = [256, 160, 128, 64, 64]
#
#             h = tf.layers.dense(units=weights[0], inputs=self.state, kernel_initializer=weights_initializer,
#                                 activation=tf.nn.relu)
#
#             for idx in range(1, number_of_layers):
#                 h = tf.layers.dense(units=weights[idx], inputs=h, kernel_initializer=weights_initializer,
#                                     activation=tf.nn.relu)
#
#             self.output = tf.layers.dense(units=action_size, inputs=h, kernel_initializer=weights_initializer,
#                                           activation=None)
#
#             self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
#             self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
#             self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
#
#             self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
#
#     def predict(self, state):
#         sess = tf.get_default_session()
#         return sess.run(self.actions_distribution, {self.state: state})
#
#     def train(self, state, td_error, action_one_hot, actor_lr):
#         sess = tf.get_default_session()
#         feed_dict_policy = {self.state: state,
#                             self.R_t: td_error,
#                             self.action: action_one_hot,
#                             self.learning_rate: actor_lr}
#         sess.run([self.optimizer, self.loss], feed_dict_policy)
#
#
# class Critic:
#     def __init__(self, state_size, learning_rate, name='critic'):
#         self.state_size = state_size
#         self.learning_rate = learning_rate
#
#         with tf.variable_scope(name):
#             self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
#             self.R_t = tf.placeholder(tf.float32, name="total_rewards")
#             self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
#
#             number_of_layers = 4
#             weights = [256, 160, 128, 64, 128]
#
#             h = tf.layers.dense(units=weights[0], inputs=self.state, kernel_initializer=weights_initializer,
#                                 activation=tf.nn.relu)
#
#             for idx in range(1, number_of_layers):
#                 h = tf.layers.dense(units=weights[idx], inputs=h, kernel_initializer=weights_initializer,
#                                     activation=tf.nn.relu)
#
#             self.output = tf.layers.dense(units=1, inputs=h, kernel_initializer=weights_initializer,
#                                           activation=None)
#
#             self.square_loss = tf.squared_difference(tf.squeeze(self.output), self.R_t)
#             self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.square_loss)
#
#     def predict(self, state):
#         sess = tf.get_default_session()
#         predict_feed_dict = {self.state: state}
#         return sess.run(self.output, {self.state: state})
#
#     def train(self, state, target, lr):
#         sess = tf.get_default_session()
#         update_feed_dict = {self.state: state, self.R_t: target, self.learning_rate: lr}
#         sess.run([self.optimizer, self.square_loss], update_feed_dict)
#

def train_actor_and_critic(actor, critic, state, next_state, done, reward, discount_factor, actor_lr, critic_lr,
                           action_one_hot):
    value = critic.predict(state)
    next_value = critic.predict(next_state)

    if done:
        td_target = reward
    else:
        td_target = reward + discount_factor * next_value

    td_error = td_target - value

    # Critic train
    # update_feed_dict = {critic.state: state, critic.R_t: td_target, critic.learning_rate: critic_lr}
    # sess.run([critic.optimizer, critic.square_loss], update_feed_dict)
    critic.train(state, td_target, critic_lr)
    # Actor train
    # feed_dict_policy = {actor.state: state,
    #                     actor.R_t: td_error,
    #                     actor.action: action_one_hot,
    #                     actor.learning_rate: actor_lr}
    # sess.run([actor.optimizer, actor.loss], feed_dict_policy)
    actor.train(state, td_error, action_one_hot, actor_lr)


# Define hyperparameters
state_size = 6
action_size = 3
actual_actions_size = env.action_space.n

max_episodes = 1500
max_steps = 501
discount_factor = 0.99
actor_lr = 0.00005
critic_lr = 0.0005
learning_rate_decay = 0.99

render = True

# Initialize the AC networks

actor = Actor(state_size, action_size, "acrobot_actor")
critic = Critic(state_size, critic_lr, "acrobot_critic")

start_time = time.time()

reward_size = 1000

# Start training the agent with REINFORCE algorithm
# summary = tf.summary.FileWriter("../tensorboard/actor_critic/acrobot", sess.graph)
# sess.run(tf.global_variables_initializer())
solved = False

# First, generate some data for initial rewards training
episode_rewards = np.zeros(max_episodes)
average_rewards = 0.0

max_iters = 2500
first = True
for episode in range(max_episodes):
    state = env.reset()
    state = np.append(state, np.zeros(6 - env.observation_space.shape[0]))
    state = state.reshape([1, state_size])

    done = False
    iter = 0

    while not done and iter < max_iters:
        actor_prediction = actor.predict(state)
        actions_distribution = tf.squeeze(tf.nn.softmax(actor_prediction)).numpy()

        action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
        while action >= actual_actions_size:
            action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
        next_state, reward, done, _ = env.step(action)
        episode_rewards[episode] += reward

        next_state = np.append(next_state, np.zeros(6 - env.observation_space.shape[0]))
        next_state = next_state.reshape([1, state_size])

        if reward == -1:
            done = False
        else:
            reward = reward_size
            max_iters = max(500, max_iters - 500)

        if render and episode % 10 == 0:
            env.render()

        action_one_hot = np.zeros(action_size)
        action_one_hot[action] = 1
        train_actor_and_critic(actor, critic, state, next_state, done, reward, discount_factor, actor_lr, critic_lr,
                               action_one_hot)
        iter += 1
        if iter == max_iters:
            done = True
        if done:
            # episode_summary = tf.Summary()
            # episode_summary.value.add(tag="Rewards", simple_value=episode_rewards[episode])
            # summary.add_summary(episode_summary, episode)

            actor_lr = actor_lr * learning_rate_decay
            critic_lr = critic_lr * learning_rate_decay

            if episode > 48:
                # Check if solved
                average_rewards = np.mean(episode_rewards[(episode - 49):episode + 1])
            print("Episode {} Reward: {} Average over 50 episodes: {}".format(episode, episode_rewards[episode],
                                                                              round(average_rewards, 2)))
            print(f"took total of {iter} iters")
            if episode_rewards[episode] > -115:
                print('Solved at episode: ' + str(episode))
                elapsed_time = time.time() - start_time
                print(f"elapsed_time: {elapsed_time}")
                solved = True
            break
        state = next_state

    if solved:
        break

plt.figure(figsize=(20, 10))
non_zero_rewards = episode_rewards[:episode + 1]
episodes = [i for i in range(len(non_zero_rewards))]
plt.plot(episodes, non_zero_rewards, color='blue', marker='o')
plt.title(f'Actor Critic algorithm. Episode till solve: {episode}', fontsize=14)
plt.xlabel('episode', fontsize=14)
plt.ylabel('Rewards', fontsize=14)
plt.grid(True)
plt.show()
