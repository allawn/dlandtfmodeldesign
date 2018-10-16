import tensorflow as tf

hidden_units = 20
cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_units)
timesteps = 5;
batch_size = 2
input = tf.placeholder(tf.float32, [batch_size, timesteps, 1], name='input_x')
outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=input, dtype=tf.float32)

