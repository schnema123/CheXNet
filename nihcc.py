import tensorflow as tf
import densenet

# Our input is a 224x224 (Or 256x256?) 8bit per Channel 1 Channel png
# => Tensor 224x224 => 
example_image = tf.random_normal([50176])

def model_fn(
    features, 
    labels, 
    mode, 
    params):
     
    in_training = (mode == tf.estimator.ModeKeys.TRAIN)
    logits = densenet_121(features, num_classes=15, is_training=in_training)

    predictions = {
        "classes": tf.argmax(input = logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions=predictions)

    # Calculate loss
    onehot_labels = tf.one_hot(indices = tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
        train_op = optimizer.minimize(
            loss = loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def input_fn():
    """An input function for training"""
    dataset = tf.data.Dataset.from_tensor(example_image)
    dataset = dataset.shuffle(1000).repeat()
    return dataset.make_one_shot_iterator().get_next()


