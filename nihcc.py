import tensorflow as tf
import densenet
import nihcc_dataset


def model_fn(
        features,
        labels,
        mode):

    tf.summary.image("image", features)

    in_training = (mode == tf.estimator.ModeKeys.TRAIN)
    net, end_points = densenet.densenet121(
        features, num_classes=15, is_training=in_training)

    logits = end_points["predictions"]
    logits = tf.reshape(logits, [-1, 15])

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    tensor_labels = tf.identity(labels, "labels")
    # Calculate loss
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=tensor_labels, logits=logits)

    tf.summary.scalar("loss", loss)


    if mode == tf.estimator.ModeKeys.TRAIN:
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


def input_fn(batch_size):
    """An input function for training"""
    ds = nihcc_dataset.create_dataset()
    ds = ds.shuffle(1000).repeat().batch(batch_size)
    return ds.make_one_shot_iterator().get_next()


def main():

    tf.logging.set_verbosity(tf.logging.INFO)

    tensors_to_log = {"probabilities": "softmax_tensor", "labels": "labels"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=2)

    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir="J:/BA/tmp/")
    estimator.train(input_fn=lambda: input_fn(5), hooks=[logging_hook])


if __name__ == "__main__":
    main()
