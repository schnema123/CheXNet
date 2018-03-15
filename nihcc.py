import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as slim_nets

import os
import densenet
import nihcc_dataset
import nihcc_utils
import nihcc_plot

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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

    tf.summary.image("image", features, max_outputs=16)

    in_training = (mode == tf.estimator.ModeKeys.TRAIN)
    net, _ = slim_nets.resnet_v2.resnet_v2_50(
        features, num_classes=14, is_training=in_training)

    logits = net
    logits = tf.reshape(logits, [-1, 14])

    probabilities = tf.sigmoid(logits, name="probabilities")
    probabilities = tf.Print(probabilities, [logits, probabilities])

    tf.identity(logits, "logits_tensor")
    tf.identity(labels, "labels_tensor")
    tf.identity(features, "image_tensor")

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

    tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits)

    print(tf.losses.get_losses())
    loss = tf.losses.get_total_loss()

    tf.identity(loss, "loss_tensor")

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


def input_fn(mode):
    """An input function for training"""
    ds = nihcc_dataset.create_dataset(mode)
    return ds.make_one_shot_iterator().get_next()


def main():

    tf.logging.set_verbosity(tf.logging.INFO)

    tensors_to_log = {"probabilities": "probabilities",
                      "labels": "labels_tensor",
                      "logits": "logits_tensor",
                      "loss": "loss_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1000)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn, model_dir="../tmp/")

    #while True:
    #    print("Training for one epoch...")
    #    estimator.train(input_fn=lambda: input_fn(tf.estimator.ModeKeys.TRAIN), hooks=[logging_hook])
    #    print("Done training.")

    #    print("Evaluating model...")
    #    eval_results = estimator.evaluate(input_fn=lambda: input_fn(tf.estimator.ModeKeys.EVAL))
    #    print(eval_results)
    #    print("Done evaluating model.")

    print("Printing ROC Curve...")
    nihcc_plot.plot_roc(input_fn, model_fn)
    print("Done printing ROC Curve")

if __name__ == "__main__":
    main()
