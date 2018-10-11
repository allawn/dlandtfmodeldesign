import tensorflow as tf
import numpy as np
import gym

env = gym.make('CartPole-v0')
state_size = 4
action_size = env.action_space.n

max_episodes = 300
learning_rate = 0.01
gamma = 0.95


def discount_and_normalize_rewards(episode_rewards):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative

    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

    return discounted_episode_rewards



input_ = tf.placeholder(tf.float32, [None, state_size], name="input_")
actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
discounted_episode_rewards_ = tf.placeholder(tf.float32, [None, ],
                                             name="discounted_episode_rewards")

mean_reward_ = tf.placeholder(tf.float32, name="mean_reward")
fc1 = tf.contrib.layers.fully_connected(inputs=input_,
                                        num_outputs=10,
                                        activation_fn=tf.nn.relu,
                                        weights_initializer=tf.contrib.layers.xavier_initializer())
fc2 = tf.contrib.layers.fully_connected(inputs=fc1,
                                        num_outputs=action_size,
                                        activation_fn=tf.nn.relu,
                                        weights_initializer=tf.contrib.layers.xavier_initializer())
fc3 = tf.contrib.layers.fully_connected(inputs=fc2,
                                        num_outputs=action_size,
                                        activation_fn=None,
                                        weights_initializer=tf.contrib.layers.xavier_initializer())
action_distribution = tf.nn.softmax(fc3)
neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc3,
                                                          labels=actions)
loss = tf.reduce_mean(neg_log_prob * discounted_episode_rewards_)
train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

allRewards = []
total_rewards = 0
maximumRewardRecorded = 0
episode = 0
episode_states, episode_actions, episode_rewards = [], [], []

saver = tf.train.Saver()

sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for episode in range(max_episodes):

    episode_rewards_sum = 0

    # Launch the game
    state = env.reset()
    while True:
        action_probability_distribution = sess.run(action_distribution, feed_dict={
            input_: state.reshape([1, 4])})

        action = np.random.choice(range(action_probability_distribution.shape[1]),
                                  p=action_probability_distribution.ravel())

        # Perform a
        env.render()
        new_state, reward, done, info = env.step(action)

        # Store s, a, r
        episode_states.append(state)

        action_ = np.zeros(action_size)
        action_[action] = 1

        episode_actions.append(action_)

        episode_rewards.append(reward)
        if done:
            # Calculate discounted reward
            discounted_episode_rewards = discount_and_normalize_rewards(
                episode_rewards)

            # Feedforward, gradient and backpropagation
            loss_, _ = sess.run([loss, train_opt], feed_dict={
                input_: np.vstack(np.array(episode_states)),
                actions: np.vstack(np.array(episode_actions)),
                discounted_episode_rewards_: discounted_episode_rewards
            })

            # Reset the transition stores
            episode_states, episode_actions, episode_rewards = [], [], []

            break

        state = new_state
