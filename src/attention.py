import tensorflow as tf


def attention(inputs, attention_size, time_major=False, return_alphas=False):
    '''
    Attention mechanism layer which reduces RNN/Bi-RNN putputs with Attention vector

    :param inputs: The Attention inputs
        Matches outputs of RNN/Bi-RNN layer (not final state):
            In case of RNN, this must be RNN outputs 'Tensor':
                If time_major == False (default), this must be a tensor of shape:
                    '[batch_size, max_time, cell.output_size]'
                Elif time_major == True, this must be a tensor of shape:
                    '[max_time, batch_size, cell.output_size]'
            In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing
            the forward and backward RNN outputs 'Tensor'\
                If time_major == False(default):
                    outputs_fw is a 'Tensor' shaped:
                    '[batch_size, max_time, cell_fw.output_size]'
                    and outputs_bw is a 'Tensor' shaped:
                    '[batch_size, max_time, cell_bw.output_size]'.
                If time_major == True:
                     outputs_fw is a 'Tensor' shaped:
                    '[max_time, batch_size, cell_fw.output_size]'
                    and outputs_bw is a 'Tensor' shaped:
                    '[max_time, batch_size, cell_bw.output_size]'.
    :param attention_size: Linear size of the Attention weights.
    :param time_major: The shape format of the 'inputs' Tensors.
        If true, these 'Tensors' must be shaped '[max_time, batch_size, depth]'
        Elif False, these 'Tensors' must be shaped '[batch_size, max_time, depth]'
        Using 'time_major == True' is a bit more efficient because it avoids transposes at
        the beginning and end of the RNN calculation. However, most TensorFlow data is batch_major,
        so by default this function accepts input and emits output in batch_major form
    :param return_alphas: Whether to return attention coefficients variable along with layer`s output.
        Used for visualization purpose.

    :return:
        The Attention output 'Tensor'.
        In case of RNN, this will be a 'Tensor' shaped:
            '[batch_size, cell.output_size]'.
        In case of Bidirectional RNN, this will be a 'Tensor' shaped；
            ‘[batch_size, cell_fw.output_size + cell_bw.output_size]’.
    '''

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and th backward RNN outputs
        inputs = tf.concat(inputs, axis=2)

    if time_major:
        # (T, B, D) => (B, T, D)
        inputs = tf.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value     # D value ---- hidden size of the RNN layer

    # Trainable parameters
    W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
    #   the shape of 'v' is (B, T, D)*(D, A), where A = attentison_size
    # v = tf.tanh(tf.tensordot(inputs, W_omega, axes=1) + b_omega)
    v = tf.sigmoid(tf.tensordot(inputs, W_omega, axes=1) + b_omega)
    # For each of the timestamps its vector of size A from 'v' is reduced with 'u' vector
    vu = tf.tensordot(v, u_omega, axes=1)   # (B, T) shape
    alphas = tf.nn.softmax(vu)      # (B, T) shape also


    # OUtput of Bi-RNN is reduced with attention vector; the result has (B, D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas