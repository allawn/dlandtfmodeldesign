import tensorflow as tf

hidden_units = 20
forwardCell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_units)
backwardCell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_units)
timesteps = 5
batch_size = 2
input = tf.placeholder(tf.float32, [batch_size, timesteps, 1], name='input_x')

outputs, output_states = tf.nn.bidirectional_dynamic_rnn(inputs=input,
                                                         cell_fw=forwardCell,
                                                         cell_bw=backwardCell,
                                                         dtype=tf.float32,
                                                         time_major=False)
