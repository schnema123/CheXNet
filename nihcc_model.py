import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as slim_nets

import nihcc_utils
import densenet
import densenet_new


def load_variables_from_checkpoint():
    checkpoint_to_load = "../imagenet_pretrained/tf-densenet121.ckpt"
    trainable_variables = tf.trainable_variables()
    assigment_map = {var.name.split(":")[0]: var for var in trainable_variables if not var.name.endswith(
        "/biases:0") and "final_block" not in var.name and "fully_connected" not in var.name}
    print(assigment_map)
    tf.train.init_from_checkpoint(
        checkpoint_to_load, assigment_map)

def network_fn(image, in_training):
    net = densenet_new.densenet121(
        image, num_classes=14, in_training=in_training)
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

    # Setup multi-gpu training
    num_gpu = 2

    global_step = tf.train.get_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, )

    image_split = tf.split(image, num_gpu)
    labels_split = tf.split(labels, num_gpu)

    tower_gradients = []

    with tf.variable_scope(tf.get_variable_scope()):
        for x in range(num_gpu):
            with tf.device("/gpu:{}".format(x)):
                with tf.name_scope("classification_{}".format(x)) as scope:

                    tf.summary.image("image", image_split[x], max_outputs=16)
                    logits = network_fn(image_split[x], True)

                    loss = loss_fn(labels_split[x], logits)

                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)
                    with tf.control_dependencies(update_ops):
                        losses = tf.get_collection(tf.GraphKeys.LOSSES, scope)
                        total_loss = tf.add_n(losses, name="total_loss")
                        tf.summary.scalar("total_loss", total_loss)

                    tf.get_variable_scope().reuse_variables()
                    gradients = optimizer.compute_gradients(total_loss)
                    tower_gradients.append(gradients)

    gradients = nihcc_utils.average_gradients(tower_gradients)
    apply_gradients_op = optimizer.apply_gradients(gradients, global_step=global_step)

    variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    
    train_op = tf.group(apply_gradients_op, variable_averages_op)
    return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)

    