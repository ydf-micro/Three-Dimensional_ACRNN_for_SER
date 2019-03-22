# -*- coding: utf-8 -*-

import tensorflow as tf
from attention import attention

epsilon = 1e-3

def weight_variable(name, shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32), name=name)


def bias_variable(name, shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1, dtype=tf.float32), name=name)


def conv2d(inputs, W, strides, padding='SAME'):
    return tf.nn.conv2d(inputs, W, strides, padding)


def max_pool(inputs, ksize, strides, padding='SAME'):
    return tf.nn.max_pool(inputs, ksize, strides, padding)


def batch_norm_wrapper(inputs, is_training, decay=0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training is not None:
        batch_mean, batch_var = tf.nn.moments(inputs, [0])  # mean, variance
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                                             batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
                                         pop_mean, pop_var, beta, scale, epsilon)


def acrnn(inputs, num_classes=4, is_training=True, in_channels=3, L1=128, L2=256, p=10, num_linear=768,
          cell_units=128, time_step=150, FC1=64, dropout_keep_prob=1):

    # variable
    layer1_filter = weight_variable(name='layer1_filter', shape=[5, 3, in_channels, L1])
    layer1_bias = bias_variable(name='layer1_bias', shape=[L1])
    layer1_strides = [1, 1, 1, 1]

    layer2_filter = weight_variable(name='layer2_filter', shape=[5, 3, L1, L2])
    layer2_bias = bias_variable(name='layer2_bias', shape=[L2])
    layer2_strides = [1, 1, 1, 1]

    layer3_filter = weight_variable(name='layer3_filter', shape=[5, 3, L2, L2])
    layer3_bias = bias_variable(name='layer3_bias', shape=[L2])
    layer3_strides = [1, 1, 1, 1]

    layer4_filter = weight_variable(name='layer4_filter', shape=[5, 3, L2, L2])
    layer4_bias = bias_variable(name='layer4_bias', shape=[L2])
    layer4_strides = [1, 1, 1, 1]

    layer5_filter = weight_variable(name='layer5_filter', shape=[5, 3, L2, L2])
    layer5_bias = bias_variable(name='layer5_bias', shape=[L2])
    layer5_strides = [1, 1, 1, 1]

    layer6_filter = weight_variable(name='layer6_filter', shape=[5, 3, L2, L2])
    layer6_bias = bias_variable(name='layer6_bias', shape=[L2])
    layer6_strides = [1, 1, 1, 1]

    linear_weight = weight_variable(name='linear_weight', shape=[p*L2, num_linear])
    linear_bias = bias_variable(name='linear_bias', shape=[num_linear])

    fully1_weight = weight_variable(name='fully1_weight', shape=[2*cell_units, FC1])
    fully1_bias = bias_variable(name='fully1_bias', shape=[FC1])

    fully2_weight = weight_variable(name='fully2_weight', shape=[FC1, num_classes])
    fully2_bias = bias_variable(name='fully2_bias', shape=[num_classes])


    # layer
    layer1 = conv2d(inputs, layer1_filter, layer1_strides, padding='SAME')
    layer1 = tf.nn.bias_add(layer1, layer1_bias)
    layer1 = tf.nn.relu(layer1)
    layer1 = tf.nn.max_pool(layer1, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID', name='max_pool')
    layer1 = tf.nn.dropout(layer1, keep_prob=dropout_keep_prob)

    layer2 = conv2d(layer1, layer2_filter, layer2_strides, padding='SAME')
    layer2 = tf.nn.bias_add(layer2, layer2_bias)
    layer2 = tf.nn.relu(layer2)
    layer2 = tf.nn.dropout(layer2, keep_prob=dropout_keep_prob)

    layer3 = conv2d(layer2, layer3_filter, layer3_strides, padding='SAME')
    layer3 = tf.nn.bias_add(layer3, layer3_bias)
    layer3 = tf.nn.relu(layer3)
    layer3 = tf.nn.dropout(layer3, keep_prob=dropout_keep_prob)

    layer4 = conv2d(layer3, layer4_filter, layer4_strides, padding='SAME')
    layer4 = tf.nn.bias_add(layer4, layer4_bias)
    layer4 = tf.nn.relu(layer4)
    layer4 = tf.nn.dropout(layer4, keep_prob=dropout_keep_prob)

    layer5 = conv2d(layer4, layer5_filter, layer5_strides, padding='SAME')
    layer5 = tf.nn.bias_add(layer5, layer5_bias)
    layer5 = tf.nn.relu(layer5)
    layer5 = tf.nn.dropout(layer5, keep_prob=dropout_keep_prob)

    layer6 = conv2d(layer5, layer6_filter, layer6_strides, padding='SAME')
    layer6 = tf.nn.bias_add(layer6, layer6_bias)
    layer6 = tf.nn.relu(layer6)
    layer6 = tf.nn.dropout(layer6, keep_prob=dropout_keep_prob)

    layer6 = tf.reshape(layer6, [-1, time_step, L2*p])
    layer6 = tf.reshape(layer6, [-1, p*L2])

    linear = tf.matmul(layer6, linear_weight) + linear_bias
    linear = batch_norm_wrapper(linear, is_training)
    linear = tf.nn.relu(linear)
    linear = tf.reshape(linear, [-1, time_step, num_linear])

    gru_fw_cell = tf.nn.rnn_cell.LSTMCell(cell_units, forget_bias=1.0, name='cell_fw')
    gru_bw_cell = tf.nn.rnn_cell.LSTMCell(cell_units, forget_bias=1.0, name='cell_bw')

    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_fw_cell,
                                                               cell_bw=gru_bw_cell,
                                                               inputs=linear,
                                                               dtype=tf.float32,
                                                               time_major=False,
                                                               scope='LSTM1')

    gru, alphas = attention(outputs, 1, return_alphas=True)

    fully1 = tf.matmul(gru, fully1_weight) + fully1_bias
    fully1 = tf.nn.relu(fully1)
    fully1 = tf.nn.dropout(fully1, keep_prob=dropout_keep_prob)

    Ylogits = tf.matmul(fully1, fully2_weight) + fully2_bias
    Ylogits = tf.nn.softmax(Ylogits)

    return Ylogits