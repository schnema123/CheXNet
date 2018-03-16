import tensorflow as tf
import os

import nihcc_input
import nihcc_model
import nihcc_utils

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():

    tf.logging.set_verbosity(tf.logging.INFO)

    tensors_to_log = {"probabilities": "probabilities",
                      "labels": "labels_tensor",
                      "logits": "logits_tensor",
                      "loss": "loss_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1000)

    estimator = tf.estimator.Estimator(
        model_fn=nihcc_model.model_fn, model_dir="../tmp/")

    while True:
        print("Training for one epoch...")
        estimator.train(input_fn=lambda: nihcc_input.input_fn(tf.estimator.ModeKeys.TRAIN), hooks=[logging_hook])
        print("Done training.")
        
        print("Evaluating model...")
        eval_results = estimator.evaluate(input_fn=lambda: nihcc_input.input_fn(tf.estimator.ModeKeys.EVAL))
        print(eval_results)
        print("Done evaluating model.")

if __name__ == "__main__":
    main()
