import nihcc_model
import nihcc_input

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn as sk
import sklearn.metrics
import numpy as np
import os


FINDINGS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
            "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
NUM_FINDINGS = len(FINDINGS)

def _plot(labels, predictions, filename):

    plt.figure(figsize=(10, 10))    

    for x in range(NUM_FINDINGS):
        label = labels[:, x]
        prediction = predictions[:, x]

        fpr, tpr, _ = sk.metrics.roc_curve(label, prediction)
        roc_auc = sk.metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label="{} (area = {:.2f})".format(FINDINGS[x], roc_auc), lw=1)

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    
    plt.savefig(filename)


def _create_or_append(arr, val):
    if arr is None:
        return val
    else:
        return np.append(arr, val, axis=0)

def plot_roc():
    
    with tf.Session() as sess:

        # Get eval dataset
        features, labels = nihcc_input.input_fn(tf.estimator.ModeKeys.PREDICT)

        # Rebuild model 
        model = nihcc_model.model_fn(features, labels, tf.estimator.ModeKeys.PREDICT)
        predictions = model.predictions

        checkpoint = "../history/dropout_0/model.ckpt-19566"
        print("Plotting checkpoint {}".format(checkpoint))

        saver = tf.train.Saver()
        saver.restore(sess, checkpoint)

        prediction_values = None
        label_values = None

        while True:
            try:
                preds, lbls = sess.run([predictions, labels])

                prediction_values = _create_or_append(prediction_values, preds["probabilities"])
                label_values = _create_or_append(label_values, lbls)

            except tf.errors.OutOfRangeError:
                break

        print("Label values size: {}".format(len(label_values)))
        print("Prediction values size: {}".format(len(prediction_values)))

        _plot(label_values, prediction_values, checkpoint + ".png")


if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    print("Printing ROC Curve...")
    plot_roc()
    print("Done printing ROC Curve")
