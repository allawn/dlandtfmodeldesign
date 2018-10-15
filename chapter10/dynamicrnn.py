import tensorflow as tf

hidden_units = 20
rnnLayerNum = 1
rnnCells = []
for i in range(rnnLayerNum):
    rnnCells.append(tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_units))

multiRnnCell = tf.nn.rnn_cell.MultiRNNCell(rnnCells)
timesteps = 5;
batch_size = 2
input = tf.placeholder(tf.float32, [batch_size, timesteps, 1], name='input_x')
sequence_length = [2, 5]
initial_state = multiRnnCell.zero_state(batch_size=2, dtype=tf.float32)

outputs, final_state = tf.nn.dynamic_rnn(multiRnnCell, input, sequence_length=sequence_length,
                                         initial_state=initial_state, dtype=tf.float32, time_major=False)

def last_output(outputs, sequence_length):
    batch_size = tf.shape(outputs)[0]
    max_length = tf.shape(outputs)[1]
    out_size = int(outputs.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (sequence_length- 1)
    flat = tf.reshape(outputs, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant
