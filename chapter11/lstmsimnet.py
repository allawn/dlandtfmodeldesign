import tensorflow as tf

batch_size = 2
timesteps = 5
embedding_size = 10
class_num = 2
num_units = 20

seq1 = tf.placeholder(tf.float32, [batch_size, timesteps, embedding_size], name='seq1')
seq2 = tf.placeholder(tf.float32, [batch_size, timesteps, embedding_size], name='seq2')
seq1Length = tf.placeholder(tf.int32, [batch_size])
seq2Length = tf.placeholder(tf.int32, [batch_size])
label_data = tf.placeholder(tf.int32, [batch_size, class_num])

cell = tf.nn.rnn_cell.LSTMCell(num_units)
outputs1, states1 = tf.nn.dynamic_rnn(cell=cell, inputs=seq1, sequence_length=seq1Length,
                                      dtype=tf.float32)
outputs2, states2 = tf.nn.dynamic_rnn(cell=cell, inputs=seq1, sequence_length=seq1Length,
                                      dtype=tf.float32)

index1 = tf.range(0, batch_size) * timesteps + (seq1Length - 1)
flat1 = tf.reshape(outputs1, [-1, num_units])
relevant1 = tf.gather(flat1, index1)

index2 = tf.range(0, batch_size) * timesteps + (seq2Length - 1)
flat2 = tf.reshape(outputs2, [-1, num_units])
relevant2 = tf.gather(flat2, index2)

concatVector = tf.concat([relevant1, relevant2], -1)
logits = tf.layers.dense(concatVector, class_num)
loss = tf.losses.softmax_cross_entropy(label_data, logits)
print(loss)
