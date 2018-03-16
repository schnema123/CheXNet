import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as slim_nets

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


def model_fn(
        features,
        labels,
        mode):

    image = features
    image = tf.reshape(image, [-1, 224, 224])
    image = tf.stack([image, image, image], axis=3)

    tf.summary.image("image", image, max_outputs=16)

    in_training = (mode == tf.estimator.ModeKeys.TRAIN)
    net = densenet_new.densenet121(
        image, num_classes=14, in_training=in_training)

    logits = net
    logits = tf.reshape(logits, [-1, 14])

    # logits = tf.Print(logits, [image], summarize=1000000, message="image")
    # logits = tf.Print(logits, [labels], summarize=1000, message="labels")
    # logits = tf.Print(logits, [logits], summarize=1000, message="logits")
    with (tf.control_dependencies([tf.add_check_numerics_ops()])):
        probabilities = tf.sigmoid(logits, name="probabilities")

    tf.identity(logits, "logits_tensor")
    tf.identity(labels, "labels_tensor")

    predictions = {
        "classes": tf.greater(probabilities, 0.5),
        "probabilities": probabilities,
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss
    # weights_constant = tf.constant([2.0] * 14, dtype=tf.float32)
    # weights_labels = tf.cast(labels, tf.float32)
    # weights = tf.multiply(weights_labels, weights_constant)

    # "Optimize the sum of unweighted binary cross entropy losses"

    tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits, reduction=tf.losses.Reduction.SUM)

    print(tf.losses.get_losses())
    loss = tf.losses.get_total_loss()
    loss = tf.identity(loss, "loss_tensor")

    tf.summary.scalar("loss", loss)
    for var in tf.global_variables():
        name = var.name.replace(":", "_")
        tf.summary.histogram(name, var)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(
            learning_rate=0.001, beta1=0.9, beta2=0.999)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"]),
        "auc": tf.metrics.auc(labels=labels, predictions=predictions["probabilities"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)