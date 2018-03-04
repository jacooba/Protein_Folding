import tensorflow as tf


def conv_layers(scope_name, inpt, channel_sizes, kernels, strides):
      with tf.variable_scope(scope_name, reuse=False):
          conv_tensor = inpt
          for channels, kernel_len, stride_len in zip(channel_sizes, kernels, strides):
              kernel = (kernel_len, kernel_len)
              stride = (1, stride_len, stride_len, 1)
              conv_tensor = tf.layers.conv2d(conv_tensor, filters=channels,
                                            kernel_size=kernel,
                                            strides=stride, padding=c.PADDING,
                                            activation=tf.nn.relu,
                                            name="%dx%d" % (kernel_len, kernel_len))
      conv_shape = conv_tensor.shape
      flat_sz = conv_shape[1].value * conv_shape[2].value * conv_shape[3].value
      flattened_conv = tf.reshape(conv_tensor, shape=[conv_shape[0].value, flat_sz])
      return flattened_conv


  def fc_layers(scope_name, inpt, layer_sizes):
      with tf.variable_scope(name, reuse=False):
          for i, sz in enumerate(layer_sizes):
              fc_layer = tf.layers.dense(inpt, sz,
                                         name='fc_%d' % i, activation=tf.nn.relu)

def calculate_entropy(p):
    return tf.reduce_sum(-tf.log(p) * p)

def unpack_batch_triple(batch_triple):
    s_imgs_batch, vectors, scalar = batch_triple
    s_vector_batch = np.concatenate(np.reshape(vectors, [-1]), scalar)
    return s_imgs_batch, s_vector_batch
