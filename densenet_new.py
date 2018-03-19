import tensorflow as tf

def _conv(x, num_filters, kernel_size, stride, in_training, dropout_rate, name):
    with tf.variable_scope(name):
        x = tf.layers.batch_normalization(x, name="batch_norm", training=in_training)
        x = tf.nn.relu(x, name="relu")
        x = tf.layers.conv2d(x, num_filters, kernel_size, stride, padding="same", name="conv")

        if dropout_rate is not None:
            x = tf.layers.dropout(x, dropout_rate)

        return x

def _conv_block(x, num_filters, kernel_size, stride, in_training, dropout_rate, name):
    with tf.variable_scope(name):
        y = x
        y = _conv(y, 4 * num_filters, 1, 1, in_training, dropout_rate, "conv_1")
        y = _conv(y,     num_filters, 3, 1, in_training, dropout_rate, "conv_2")
        y = tf.concat([x, y], axis=3)
        return y

def _dense_block(x, num_layers, num_filters, growth_rate, in_training, dropout_rate, name):
    with tf.variable_scope(name):
        for i in range(num_layers):
            x = _conv_block(x, growth_rate, 1, 1, in_training, dropout_rate, "conv_block" + str(i))
            num_filters += growth_rate
        return x, num_filters

def _transition_block(x, num_filters, compression, in_training, dropout_rate, name):
    with tf.variable_scope(name):
        num_filters = num_filters * compression
        x = _conv(x, num_filters, 1, 1, in_training, dropout_rate, "conv")
        x = tf.layers.average_pooling2d(x, 2, 2, padding="same", name="avg_pool")
        return x, num_filters

def densenet121(inputs, num_classes,
    num_filters = 64,
    growth_rate = 32,
    compression = 0.5,
    dropout_rate = 0.5,
    in_training = True,
    name="densenet121"):

    with tf.variable_scope(name):
        net = inputs
        
        # Initial convolution
        net = tf.layers.conv2d(net, num_filters, 7, 2, padding="same", name="initial_conv")
        net = tf.layers.batch_normalization(net, name="batch_norm", training=in_training)
        net = tf.nn.relu(net, name="relu")

        # Max Pooling
        net = tf.layers.max_pooling2d(net, 3, 2, padding="same")

        # Dense blocks
        # 6 -> 12 -> 24 -> 16

        net, num_filters = _dense_block(net, 6, num_filters, growth_rate, in_training, dropout_rate, "dense_block1")
        net, num_filters = _transition_block(net, num_filters, compression, in_training, dropout_rate, "transition_block1")

        net, num_filters = _dense_block(net, 12, num_filters, growth_rate, in_training, dropout_rate, "dense_block2")
        net, num_filters = _transition_block(net, num_filters, compression, in_training, dropout_rate, "transition_block2")

        net, num_filters = _dense_block(net, 24, num_filters, growth_rate, in_training, dropout_rate, "dense_block3")
        net, num_filters = _transition_block(net, num_filters, compression, in_training, dropout_rate, "transition_block3")

        net, num_filters = _dense_block(net, 16, num_filters, growth_rate, in_training, dropout_rate, "dense_block4")

        net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)

        # Do global average pooling
        net = tf.reduce_mean(net, [1, 2], name="global_avg_pool")

        # TODO: Remove this?
        net = tf.contrib.layers.flatten(net)
        # TODO: Activation function?
        net = tf.layers.dense(net, num_classes)

        return net
