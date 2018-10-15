import tensorflow as tf

def static_bi_rnn(inputs, lengths, num_units, num_layers, dropout):
    cell = tf.nn.rnn_cell.BasicRNNCell
    cells_fw = [cell(num_units) for _ in range(num_layers)]
    cells_bw = [cell(num_units) for _ in range(num_layers)]
    if dropout > 0.0:
        cells_fw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_fw]
        cells_bw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_bw]
    outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        cells_fw=cells_fw,
        cells_bw=cells_bw,
        inputs=inputs,
        sequence_length=lengths,
        dtype=tf.float32,
        scope="birnn")
    return outputs
