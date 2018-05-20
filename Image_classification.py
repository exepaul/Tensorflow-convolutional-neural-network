import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


input_x=tf.placeholder(shape=[None,784],dtype=tf.float32)
labels=tf.placeholder(shape=[None,10],dtype=tf.float32)


def con2dlayer(x, w, b, strip_s=1):
    con2dlayer = tf.nn.conv2d(x, w, strides=[1, strip_s, strip_s, 1], padding='SAME')
    con2dlayer = tf.nn.bias_add(con2dlayer, b)
    return tf.nn.relu(con2dlayer)


def max_pool(x, k=2):
    max_pool_layer = tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    return max_pool_layer


def convo_network(x, num_classes):
    weight = tf.get_variable(name='weight0', shape=[5, 5, 1, 32],
                             initializer=tf.random_uniform_initializer(-0.01, 0.01), dtype=tf.float32)
    bias0 = tf.get_variable(name='bias0', shape=[32], initializer=tf.random_uniform_initializer(-0.01, 0.01),
                            dtype=tf.float32)

    xi = tf.reshape(x, [-1, 28, 28, 1])
    con_layer = con2dlayer(xi, weight, bias0)

    max_pool_zero = max_pool(con_layer)

    weight1 = tf.get_variable(name='weight1', shape=[5, 5, 32, 64],
                              initializer=tf.random_uniform_initializer(-0.01, 0.01), dtype=tf.float32)
    bias = tf.get_variable(name='bias1', shape=[64], initializer=tf.random_uniform_initializer(-0.01, 0.01),
                           dtype=tf.float32)

    con_2_layer = con2dlayer(max_pool_zero, weight1, bias)
    max_pool_one = max_pool(con_2_layer)

    fc_1 = tf.get_variable(name='fc1', shape=[7 * 7 * 64, 1024], initializer=tf.random_uniform_initializer(-0.01, 0.01),
                           dtype=tf.float32)
    bias_1 = tf.get_variable(name='bias', shape=[1024], initializer=tf.random_uniform_initializer(-0.01, 0.01),
                             dtype=tf.float32)

    result = tf.reshape(max_pool_one, [-1, fc_1.get_shape().as_list()[0]])

    out1 = tf.add(tf.matmul(result, fc_1), bias_1)

    fc_2 = tf.nn.relu(out1)

    droput1 = tf.nn.dropout(fc_2, 0.5)

    fc2 = tf.get_variable(name='fc2', shape=[1024, num_classes], initializer=tf.random_uniform_initializer(-0.01, 0.01),
                          dtype=tf.float32)

    bias_2 = tf.get_variable(name='bias2', shape=[num_classes], initializer=tf.random_uniform_initializer(-0.01, 0.01),
                             dtype=tf.float32)

    final_output = tf.add(tf.matmul(droput1, fc2), bias_2)

    return final_output


logits = convo_network(input_x, 10)
# cross_entropy
ce = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
loss = tf.reduce_mean(ce)

# pred
pred = tf.nn.softmax(logits)
prob = tf.argmax(pred, axis=-1)

# accuracy
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
train = tf.train.AdamOptimizer(0.001).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(200):
        batch_x, batch_y = mnist.train.next_batch(128)
        loso,acc=sess.run([loss,train], feed_dict={input_x: batch_x, labels: batch_y})
        print("loss",loso)
        print("accuracy",sess.run(accuracy, feed_dict={input_x: mnist.test.images[:256],
                                      labels: mnist.test.labels[:256]}))
