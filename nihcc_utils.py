import tensorflow as tf


class CheckPointSaverListener(tf.train.CheckpointSaverListener):

    def __init__(self, estimator, input_fn):
        self.estimator = estimator
        self.input_fn = input_fn

    def after_save(self, session, global_step):

        print("CheckpointerSaverListener: Saved CP on global_step {}".format(global_step))

        # Evaluate model for this checkpoint
        # Create and save ROC-Curve
        # Calculate AUC

