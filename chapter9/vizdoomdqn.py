import tensorflow as tf
import numpy as np
from vizdoom import *
import random
from skimage import transform
from collections import deque


def create_environment():
    game = DoomGame()
    game.load_config("basic.cfg")
    game.set_doom_scenario_path("basic.wad")
    game.init()
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]
    return game, possible_actions


game, possible_actions = create_environment()

state_size = [84, 84, 4]
action_size = game.get_available_buttons_size()
learning_rate = 0.0002

total_episodes = 5000
max_steps = 100
batch_size = 64

explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.0001

gamma = 0.99

pretrain_length = batch_size
memory_size = 50000

stack_size = 4

training = True


def preprocess_frame(frame):
    cropped_frame = frame[30:-10, 30:-30]
    normalized_frame = cropped_frame / 255.0
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])
    return preprocessed_frame


stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)],
                       maxlen=4)


def stack_frames(stacked_frames, state):
    frame = preprocess_frame(state)
    stacked_frames.append(frame)
    stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state


class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, 3], name="actions_")
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                          filters=32,
                                          kernel_size=[8, 8],
                                          strides=[4, 4],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1")

            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                                 training=True,
                                                                 epsilon=1e-5,
                                                                 name='batch_norm1')

            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")

            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                          filters=64,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2")

            self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                                 training=True,
                                                                 epsilon=1e-5,
                                                                 name='batch_norm2')

            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")

            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                          filters=128,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")

            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                                 training=True,
                                                                 epsilon=1e-5,
                                                                 name='batch_norm3')

            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")

            self.flatten = tf.layers.flatten(self.conv3_out)

            self.fc = tf.layers.dense(inputs=self.flatten,
                                      units=512,
                                      activation=tf.nn.elu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="fc1")

            self.output = tf.layers.dense(inputs=self.fc,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units=3,
                                          activation=None)

            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(
                self.loss)


tf.reset_default_graph()
DQNetwork = DQNetwork(state_size, action_size, learning_rate)


class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)
        return [self.buffer[i] for i in index]


game.new_episode()
memory = Memory(max_size=memory_size)

for i in range(pretrain_length):
    if i == 0:
        state = game.get_state().screen_buffer
        state = stack_frames(stacked_frames, state)
    action = random.choice(possible_actions)
    reward = game.make_action(action)
    done = game.is_episode_finished()

    if done:
        next_state = np.zeros(state.shape)
        memory.add((state, action, reward, next_state, done))
        game.new_episode()
    else:
        next_state = game.get_state().screen_buffer
        next_state = stack_frames(stacked_frames, next_state)
        memory.add((state, action, reward, next_state, done))
        state = next_state

rewards_list = []

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
game.init()
decay_step = 0

for episode in range(total_episodes):
    game.new_episode()
    step = 0
    frame = game.get_state().screen_buffer
    state = stack_frames(stacked_frames, frame)

    while step < max_steps:
        step += 1
        decay_step += 1
        exp_exp_tradeoff = np.random.rand()

        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(
            -decay_rate * decay_step)

        if (explore_probability > exp_exp_tradeoff):
            action = random.choice(possible_actions)

        else:
            Qs = sess.run(DQNetwork.output, feed_dict={
                DQNetwork.inputs_: state.reshape((1, *state.shape))})
            action = np.argmax(Qs)
            action = possible_actions[int(action)]

        reward = game.make_action(action)
        done = game.is_episode_finished()

        if done:
            next_state = np.zeros((84, 84), dtype=np.int)
            next_state = stack_frames(stacked_frames, next_state)

            step = max_steps

            total_reward = game.get_total_reward()

            print('Episode: {}'.format(episode),
                  'Total reward: {}'.format(total_reward),
                  'Training loss: {:.4f}'.format(loss),
                  'Explore P: {:.4f}'.format(explore_probability))

            rewards_list.append((episode, total_reward))
            memory.add((state, action, reward, next_state, done))
        else:
            next_state = game.get_state().screen_buffer
            next_state = stack_frames(stacked_frames, next_state)
            memory.add((state, action, reward, next_state, done))
            state = next_state

        batch = memory.sample(batch_size)
        states = np.array([each[0] for each in batch], ndmin=3)
        actions = np.array([each[1] for each in batch])
        rewards = np.array([each[2] for each in batch])
        next_states = np.array([each[3] for each in batch])
        dones = np.array([each[4] for each in batch])

        target_Qs_batch = []
        target_Qs = sess.run(DQNetwork.output,
                             feed_dict={DQNetwork.inputs_: next_states})

        for i in range(0, len(batch)):
            terminal = dones[i]
            if terminal:
                target_Qs_batch.append(rewards[i])
            else:
                target = rewards[i] + gamma * np.max(target_Qs[i])
                target_Qs_batch.append(target)

        targets = np.array([each for each in target_Qs_batch])

        loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                           feed_dict={DQNetwork.inputs_: states,
                                      DQNetwork.target_Q: targets,
                                      DQNetwork.actions_: actions})
