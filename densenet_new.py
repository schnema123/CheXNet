import tensorflow as tf

def _conv(x, num_filters, kernel_size, stride, in_training, name):
    with tf.variable_scope(name):
        x = tf.layers.batch_normalization(x, name="batch_norm", training=in_training)
        x = tf.nn.relu(x, name="relu")
        x = tf.layers.conv2d(x, num_filters, kernel_size, stride, padding="same", name="conv")
        return x

def _conv_block(x, num_filters, kernel_size, stride, in_training, name):
    with tf.variable_scope(name):
        y = x
        y = _conv(x, 4 * num_filters, 1, 1, in_training, "conv_1")
        y = _conv(x,     num_filters, 3, 1, in_training, "conv_2")
        y = tf.concat([x, y], axis=3)
        return y

def _dense_block(x, num_layers, num_filters, growth_rate, in_training, name):
    with tf.variable_scope(name):
        for i in range(num_layers):
            x = _conv_block(x, num_filters, 1, 1, in_training, "conv_block" + str(i))
            num_filters += growth_rate
        return x, num_filters

def _transition_block(x, num_filters, compression, in_training, name):
    with tf.variable_scope(name):
        num_filters = num_filters * compression
        x = _conv(x, num_filters, 1, 1, in_training, "conv")
        x = tf.layers.average_pooling2d(x, 2, 2, padding="same", name="avg_pool")
        return x, num_filters

def densenet121(inputs, num_classes,
    num_filters = 64,
    growth_rate = 32,
    compression = 0.5,
    in_training = True,
    name="densenet121"):

    with tf.variable_scope(name):
        # Initial convolution
        net = inputs
        
        net = tf.layers.conv2d(net, num_filters, 7, 2, padding="same", name="initial_conv")
        net = tf.layers.batch_normalization(net, name="batch_norm", training=in_training)
        net = tf.nn.relu(net, name="relu")

        # Max Pooling
        net = tf.layers.max_pooling2d(net, 3, 2, padding="same")

        # Dense blocks
        # 6 -> 12 -> 24 -> 16

        # TODO (Possible bug!): Should we not save the 

        net, num_filters = _dense_block(net, 6, num_filters, growth_rate, in_training, "dense_block1")
        net, num_filters = _transition_block(net, num_filters, compression, in_training, "transition_block1")

        net, num_filters = _dense_block(net, 12, num_filters, growth_rate, in_training, "dense_block2")
        net, num_filters = _transition_block(net, num_filters, compression, in_training, "transition_block2")

        net, num_filters = _dense_block(net, 24, num_filters, growth_rate, in_training, "dense_block3")
        net, num_filters = _transition_block(net, num_filters, compression, in_training, "transition_block3")

        net, num_filters = _dense_block(net, 16, num_filters, growth_rate, in_training, "dense_block4")

        # TODO (Possible bug!): Add batch_norm + relu here?

        # Do global average pooling
        net = tf.reduce_mean(net, [1, 2], name="global_avg_pool")

        # TODO: Remove this?
        net = tf.layers.flatten(net)
        # TODO: Activation function?
        net = tf.layers.dense(net, num_classes)

        return net
