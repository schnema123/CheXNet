import tensorflow as tf
import densenet
import nihcc_dataset

slim = tf.contrib.slim


def model_fn(
        features,
        labels,
        mode):

    tf.summary.image("image", features, max_outputs=10)

    in_training = (mode == tf.estimator.ModeKeys.TRAIN)
    net, _ = densenet.densenet121(
        features, num_classes=14, is_training=in_training)

    logits = net
    logits = tf.reshape(logits, [-1, 14])

    probabilities = tf.sigmoid(logits, name="probabilities")

    tf.identity(logits, "logits_tensor")
    tf.identity(labels, "labels_tensor")
    tf.identity(features, "image_tensor")

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": probabilities
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=probabilities)

    # Calculate loss
    loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits)

    tf.identity(loss, "loss_tensor")

    tf.summary.scalar("loss", loss)
    for var in tf.global_variables():
        tf.summary.histogram(var.name, var)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # optimizer = tf.train.AdamOptimizer(
        #    learning_rate=0.001, beta1=0.9, beta2=0.999)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def input_fn(batch_size, mode):
    """An input function for training"""
    # TODO: Do not shuffle and repeat when doing evaluation?
    ds = nihcc_dataset.create_dataset(mode)
    ds = ds.apply(tf.contrib.data.shuffle_and_repeat(100))
    ds = ds.batch(batch_size)
    return ds.make_one_shot_iterator().get_next()


def main():

    tf.logging.set_verbosity(tf.logging.INFO)

    tensors_to_log = {"probabilities": "probabilities",
                      "labels": "labels_tensor",
                      "logits": "logits_tensor",
                      "loss": "loss_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn, model_dir="J:/BA/tmp/")
    estimator.train(input_fn=lambda: input_fn(
        16, tf.estimator.ModeKeys.TRAIN), hooks=[logging_hook])


if __name__ == "__main__":
    main()
