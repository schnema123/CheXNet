import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.examples.tutorials.mnist.input_data as input_data

def create_net(inputs):

    with slim.arg_scope([slim.fully_connected],
        activation_fn=tf.nn.relu,
        weights_initializer=tf.truncated_normal_initializer(),
        weights_regularizer=slim.l2_regularizer(0.0005)):

        # Input layer
        net = slim.fully_connected(inputs, 150, scope='input')
        net = slim.repeat(net, 2, slim.fully_connected, 150, scope='hidden')
        net = slim.fully_connected(net, 10, scope='logits')

        return net

def load_mnist():

    mnist = input_data.read_data_sets("./tmp/", one_hot=True)
    return mnist

mnist = load_mnist()
images = mnist.train.images
labels = mnist.train.labels.astype(np.dtype('float32'))

predictions = create_net(images)
print(images)

print(slim.model_analyzer.tensor_description(predictions))

tf.losses.softmax_cross_entropy(predictions, labels)

print(predictions)
print(labels)
total_loss = tf.losses.get_total_loss()
tf.summary.scalar('losses/total_loss', total_loss)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

train_tensor = slim.learning.create_train_op(total_loss, optimizer)

train_log_dir = "./log"
slim.learning.train(train_tensor, train_log_dir)

