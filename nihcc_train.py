import tensorflow as tf
import tensorflow.contrib.keras
import os
import datetime

import nihcc_input
import nihcc_model
import nihcc_utils

def binary_crossentropy(y_true, y_pred):
    return tf.losses.sigmoid_cross_entropy(multi_class_labels=y_true, logits=y_pred, reduction=tf.losses.Reduction.SUM)

def main():

    tf.logging.set_verbosity(tf.logging.INFO)

    tensors_to_log = {"probabilities": "probabilities",
                      "labels": "labels_tensor",
                      "logits": "logits_tensor",
                      "loss": "loss_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1000)

    run_config = tf.estimator.RunConfig()
    run_config = run_config.replace(
        keep_checkpoint_max=None,
        save_checkpoints_steps=None,
        save_checkpoints_secs=60*60 # Save one checkpoint every hour
    )
    
    estimator = tf.estimator.Estimator(
      model_fn=nihcc_model.model_fn, model_dir="../tmp/", config=run_config)

    #  densenet121 = tf.keras.applications.DenseNet121(classes=14, weights=None)
    #  densenet121.compile(optimizer="adam", loss=binary_crossentropy)
    #  estimator = tf.keras.estimator.model_to_estimator(densenet121, model_dir="../tmp/")

    train_for_n_epochs = 100
    current_epoch = 0

    # while current_epoch < train_for_n_epochs:
    while True:
        print("Training for 2 epochs...")
        estimator.train(input_fn=lambda: nihcc_input.input_fn(tf.estimator.ModeKeys.TRAIN))
        print("Done training.")
        
        print("Evaluating model...")
        eval_results = estimator.evaluate(input_fn=lambda: nihcc_input.input_fn(tf.estimator.ModeKeys.EVAL))
        print(eval_results)
        print("Done evaluating model.")

        print("Done with epoch {}".format(current_epoch))
        current_epoch = current_epoch + 1

if __name__ == "__main__":
    main()
