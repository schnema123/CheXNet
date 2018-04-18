import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as slim_nets


import nets.nets_factory as densenet_factory
import nihcc_utils
import densenet_new


def load_variables_from_checkpoint():
    checkpoint_to_load = "../imagenet_pretrained/tf-densenet121.ckpt"
    trainable_variables = tf.trainable_variables()

    assigment_map = {var.name.split(":")[0]: var for var in trainable_variables
                     if not var.name.endswith("/biases:0") and
                     "/logits/" not in var.name
                     }

    tf.train.init_from_checkpoint(
        checkpoint_to_load, assigment_map)


def network_fn(image, in_training):
    net_func = densenet_factory.get_network_fn(
        "densenet121", 14, data_format="NCHW", is_training=in_training)
    net, end_points = net_func(image)
    net = tf.reshape(net, (-1, 14))

    load_variables_from_checkpoint()

    return net


def loss_fn(labels, logits):
    return tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits, reduction=tf.losses.Reduction.MEAN)


def model_fn(
        features,
        labels,
        mode):

    image = features["input_1"]

    if mode == tf.estimator.ModeKeys.PREDICT or mode == tf.estimator.ModeKeys.EVAL:

        logits = network_fn(image, False)

        with (tf.control_dependencies([tf.add_check_numerics_ops()])):
            probabilities = tf.sigmoid(logits, name="probabilities")

        classes = tf.greater(probabilities, 0.5)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                "classes": classes,
                "probabilities": probabilities,
            }
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(labels=labels, predictions=classes),
                "auc": tf.metrics.auc(labels=labels, predictions=probabilities)
            }
            loss = loss_fn(labels, logits)
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    tf.summary.image("image", image, max_outputs=16)
    logits = network_fn(image, True)

    loss = loss_fn(labels, logits)

    # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = loss  # + tf.add_n(reg_losses)
    tf.summary.scalar("total_loss", total_loss)

    global_step = tf.train.get_global_step()
    optimizer = tf.train.AdamOptimizer(
        learning_rate=0.001, beta1=0.9, beta2=0.999)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        minimize_op = optimizer.minimize(total_loss, global_step=global_step)

    variable_averages = tf.train.ExponentialMovingAverage(
        0.9999, num_updates=global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    train_op = tf.group(minimize_op, variable_averages_op)
    return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)
