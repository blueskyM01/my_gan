import tensorflow as tf


def m4_leak_relu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def m4_active_function(input_, active_function='relu', name='m4_active_function'):
    with tf.variable_scope(name):
        if active_function == 'relu':
            active = tf.nn.relu(input_)
        elif active_function == 'leak_relu':
            active = m4_leak_relu(input_)
        return active


def m4_conv_moudel(input_, fiters, k_h=3, k_w=3, s_h=3, s_w=3, stddev=0.02, padding="SAME", active_function='relu',
                   norm='batch_norm',is_trainable=True, do_active=True, name='m4_conv_moudel'):
    with tf.variable_scope(name):
        conv = m4_conv(input_, fiters, k_h, k_w, s_h, s_w, padding, stddev)
        if do_active:
            conv = m4_active_function(conv, active_function)
        if norm == 'batch_norm':
            conv = tf.contrib.layers.batch_norm(conv, decay=0.9,
                                                updates_collections=None,
                                                epsilon=1e-5,
                                                scale=True,
                                                is_training=is_trainable)
        return conv


def m4_conv(input_, fiters, k_h=3, k_w=3, s_h=3, s_w=3, padding="SAME", stddev=0.02, name='m4_conv'):
    with tf.variable_scope(name):
        batch, heigt, width, nc = input_.get_shape().as_list()
        w = tf.get_variable('w', [k_h, k_w, nc, fiters],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        bias = tf.get_variable('bias', [fiters], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_, w, strides=[1, s_h, s_w, 1], padding=padding) + bias
        return conv


def m4_deconv_moudel(input_, fiters, k_h=3, k_w=3, s_h=3, s_w=3, padding="SAME", stddev=0.02, active_function='relu',
                   norm='batch_norm',is_trainable=True, do_active=True,name='m4_deconv_moudel'):
    with tf.variable_scope(name):
        deconv = m4_deconv(input_, output_shape, k_h, k_w, s_h, s_w, padding, stddev)
        if do_active:
            deconv = m4_active_function(deconv, active_function)
        if norm == 'batch_norm':
            deconv = tf.contrib.layers.batch_norm(deconv, decay=0.9,
                                                updates_collections=None,
                                                epsilon=1e-5,
                                                scale=True,
                                                is_training=is_trainable)
        return deconv



def m4_deconv(input_, output_shape, k_h=3, k_w=3, s_h=3, s_w=3, padding="SAME", stddev=0.02, name='m4_deconv'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, s_h, s_w, 1])

            # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, s_h, s_w, 1])
        deconv = deconv + biases
        return deconv


def m4_average_grads(tower):
    averaged_grads = []
    for grads_and_vars in zip(*tower):
        # print(grads_and_vars)
        grads = []
        for g, _ in grads_and_vars:
            expanded_grad = tf.expand_dims(g, 0, 'expand_grads')
            grads.append(expanded_grad)
        grad = tf.concat(values=grads, axis=0)
        grad = tf.reduce_mean(input_tensor=grad, axis=0, keepdims=False)
        g_and_v = (grad, grads_and_vars[0][1])
        averaged_grads.append(g_and_v)
    return averaged_grads




