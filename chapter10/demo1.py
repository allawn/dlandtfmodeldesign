import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

training_examples = 10000
testing_examples = 1000
sample_gap = 0.01
timesteps = 5


def generate_data(seq):
    X = []
    y = []
    for i in range(len(seq) - timesteps - 1):
        X.append(seq[i: i + timesteps])
        y.append(seq[i + timesteps])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


test_start = training_examples * sample_gap
test_end = test_start + testing_examples * sample_gap

train_x, train_y = generate_data(np.sin(np.linspace(0, test_start, training_examples)))
test_x, test_y = generate_data(
    np.sin(np.linspace(test_start, test_end, testing_examples)))

lstm_size = 30
lstm_layers = 2
batch_size = 64

x = tf.placeholder(tf.float32, [None, timesteps, 1], name='input_x')
y_ = tf.placeholder(tf.float32, [None, 1], name='input_y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)


def lstm_cell():
    return tf.contrib.rnn.BasicLSTMCell(lstm_size)


cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(lstm_layers)])

outputs, final_state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
outputs = outputs[:, -1]

predictions = tf.contrib.layers.fully_connected(outputs, 1, activation_fn=tf.tanh)
cost = tf.losses.mean_squared_error(y_, predictions)
optimizer = tf.train.AdamOptimizer().minimize(cost)


def get_batches(X, y, batch_size=64):
    for i in range(0, len(X), batch_size):
        begin_i = i
        end_i = i + batch_size if (i + batch_size) < len(X) else len(X)
        yield X[begin_i:end_i], y[begin_i:end_i]


epochs = 20
session = tf.Session()
with session.as_default() as sess:
    tf.global_variables_initializer().run()

    iteration = 1

    for e in range(epochs):
        for xs, ys in get_batches(train_x, train_y, batch_size):
            feed_dict = {x: xs[:, :, None], y_: ys[:, None], keep_prob: .5}

            loss, _ = sess.run([cost, optimizer], feed_dict=feed_dict)

            if iteration % 100 == 0:
                print('Epochs:{}/{}'.format(e, epochs),
                      'Iteration:{}'.format(iteration),
                      'loss value: {:.8f}'.format(loss))
            iteration += 1

with session.as_default() as sess:
    feed_dict = {x: test_x[:, :, None], keep_prob: 1.0}
    results = sess.run(predictions, feed_dict=feed_dict)
    plt.plot(results, 'r', label='predict value')
    plt.plot(test_y, 'g--', label='target value')
    plt.legend()
    plt.show()
