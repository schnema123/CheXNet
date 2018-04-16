import tensorflow as tf


def _conv2d(x, num_filters, kernel_size, stride):
    """ Create one convolutional layer """
    return tf.layers.conv2d(inputs=x, filters=num_filters,
                            kernel_size=kernel_size,
                            strides=stride,
                            padding="same", name="conv2d",
                            use_bias=False, 
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1e-6))


def _batch_norm(x, in_training):
    return tf.layers.batch_normalization(
            x, name="batch_norm", training=in_training)


def _conv(x, num_filters, kernel_size, stride, in_training, dropout_rate, name):

    with tf.variable_scope(name):

        x = _batch_norm(x, in_training=in_training)
        x = tf.nn.relu(x, name="relu")
        x = _conv2d(x, num_filters=num_filters,
                    kernel_size=kernel_size, stride=stride)

        if dropout_rate is not None:
            x = tf.layers.dropout(x, rate=dropout_rate, training=in_training)

        return x


def _conv_block(x, num_filters, in_training, dropout_rate, name):
    with tf.variable_scope(name):
        y = x

        bn_num_filters = 4 * num_filters
        y = _conv(y, num_filters=bn_num_filters, kernel_size=1, stride=1,
                  in_training=in_training, dropout_rate=dropout_rate, name="conv_1")

        y = _conv(y, num_filters=num_filters, kernel_size=3, stride=1,
                  in_training=in_training, dropout_rate=dropout_rate, name="conv_2")

        y = tf.concat([x, y], axis=3)
        return y


def _dense_block(x, num_layers, num_filters, growth_rate, in_training, dropout_rate, name):
    with tf.variable_scope(name):
        for i in range(num_layers):

            x = _conv_block(x, num_filters=growth_rate, in_training=in_training,
                            dropout_rate=dropout_rate, name="conv_block" + str(i))

            num_filters += growth_rate

        return x, num_filters


def _transition_block(x, num_filters, compression, in_training, dropout_rate, name):
    with tf.variable_scope(name):

        num_filters = num_filters * compression
        x = _conv(x, num_filters=num_filters, kernel_size=1, stride=1,
                  in_training=in_training, dropout_rate=dropout_rate, name="conv")
        x = tf.layers.average_pooling2d(
            x, pool_size=2, strides=2, padding="same", name="avg_pool")

        return x, num_filters


def densenet121(inputs, num_classes,
                num_filters=64,
                growth_rate=32,
                compression=0.5,
                dropout_rate=None,
                in_training=True,
                name="densenet121"):

    with tf.variable_scope(name):
        net = inputs

        # Initial convolution
        with tf.variable_scope("initial_conv"):
            net = _conv2d(net, num_filters, kernel_size=7, stride=2)
            net = _batch_norm(net, in_training=in_training)
            net = tf.nn.relu(net, name="relu")

        # Max Pooling
        net = tf.layers.max_pooling2d(
            net, pool_size=3, strides=2, padding="same")

        # Dense blocks
        # 6 -> 12 -> 24 -> 16

        block_config = (6, 12, 24, 16)
        for i, num_layers in enumerate(block_config):

            net, num_filters = _dense_block(
                net, num_layers, num_filters, growth_rate, in_training, dropout_rate, "dense_block{}".format(i))
            
            if i != len(block_config) - 1:
                net, num_filters = _transition_block(
                    net, num_filters, compression, in_training, dropout_rate, "transition_block{}".format(i))

        net = _batch_norm(net, in_training=in_training)
        net = tf.nn.relu(net)

        # Do global average pooling
        net = tf.reduce_mean(net, [1, 2], name="global_avg_pool")
        # net = tf.layers.average_pooling2d(net, pool_size=7, strides=1, name="global_avg_pool")

        # TODO: Remove this?
        # net = tf.contrib.layers.flatten(net)
        # TODO: Activation function?
        net = tf.layers.dense(net, num_classes)

        return net
