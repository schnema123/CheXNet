import nihcc_model
import nihcc_input

import tensorflow as tf
import sklearn as sk
import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np

def _plot(labels, predictions, filename):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    print(labels)
    print(predictions)

    n_classes = 14
    for x in range(n_classes):
        label = labels[:, x]
        prediction = predictions[:, x]

        print("Printing class {}".format(x))
        print(label)
        print(prediction)

        fpr[x], tpr[x], _ = sk.metrics.roc_curve(label, prediction)
        roc_auc[x] = sk.metrics.auc(fpr[x], tpr[x])

    plt.figure()

    lw = 2
    
    for x in range(n_classes):
        plt.plot(fpr[x], tpr[x], label='ROC curve (area = %0.2f)' % roc_auc[x], lw=lw)

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    
    plt.savefig("../tmp/" + filename)


def _create_or_append(arr, val):
    if arr is None:
        return val
    else:
        return np.append(arr, val, axis=0)

def plot_roc():
    
    # Get eval dataset
    features, labels = nihcc_input.input_fn(tf.estimator.ModeKeys.EVAL)

    # Rebuild model
    model = nihcc_model.model_fn(features, labels, tf.estimator.ModeKeys.PREDICT)
    predictions = model.predictions

    saver = tf.train.Saver()
    with tf.Session() as sess:

        checkpoint = tf.train.latest_checkpoint("../tmp/")
        saver.restore(sess, checkpoint)

        prediction_values = None
        label_values = None
        batch_nr = 0

        while True:
            try:
                preds, lbls = sess.run([predictions, labels])

                prediction_values = _create_or_append(prediction_values, preds["probabilities"])
                label_values = _create_or_append(label_values, lbls)

                batch_nr = batch_nr + 1
                print("Done with batch no. {}".format(batch_nr))

            except tf.errors.OutOfRangeError:
                break

        _plot(label_values, prediction_values, checkpoint + ".png")


print("Printing ROC Curve...")
plot_roc()
print("Done printing ROC Curve")
