import tensorflow as tf
import utils
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
X_train, y_train, X_test, y_test, X_miss, y_miss, index_data = utils.get_data()
print(index_data)
y_train = np.array(pd.get_dummies(pd.DataFrame(y_train).iloc[:, 0]))
y_test = np.array(pd.get_dummies(pd.DataFrame(y_test).iloc[:, 0]))
y_miss = np.array(pd.get_dummies(pd.DataFrame(y_miss).iloc[:, 0]))

def basicNN():
    #  def nn layer
    def add_layer(inputs, in_size, out_size, activation_function=None):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size])) + 0.1
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function == None:
            output = Wx_plus_b
        else:
            output = activation_function(Wx_plus_b)
        return output



    # set up placeholder
    xs = tf.placeholder(tf.float32, [None, 56], name="features")
    ys = tf.placeholder(tf.float32, [None, 6], name="targets")

    # set up structure
    l1 = add_layer(xs, 56, 512, activation_function=tf.nn.sigmoid)  # relu --> cross
    l2 = add_layer(l1, 512, 64, activation_function=tf.nn.sigmoid)
    # l3 = add_layer(l2, 512, 256, activation_function=tf.nn.relu)
    # l4 = add_layer(l3, 256, 128, activation_function=tf.nn.relu)
    # l5 = add_layer(l4, 128, 64, activation_function=tf.nn.relu)
    prediction = add_layer(l2, 64, 6, activation_function=tf.nn.relu)
    prediction_ = tf.nn.softmax(prediction)
    print(prediction_)
    # loss function
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys))
    cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=prediction, targets=ys, pos_weight=np.array([2.,200.,2000.,200000.,2000000.,20000.])))

    train_step = tf.train.AdamOptimizer(0.0001).minimize(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in range(0, 100000):
        sess.run(train_step, feed_dict={xs: X_train, ys: y_train})
        if i % 20 == 0:
            print('process {0}'.format(i))
            print('Train Accuracy:', sess.run(accuracy, feed_dict={xs: X_train, ys: y_train}))
            print('Test Accuracy:', sess.run(accuracy, feed_dict={xs: X_test, ys: y_test}))
            print('Miss Accuracy:', sess.run(accuracy, feed_dict={xs: X_miss, ys: y_miss}))
            print('cost:', sess.run(cost, feed_dict={xs: X_train, ys: y_train}))
            a = np.array(pd.DataFrame(y_train).idxmax(axis=1))
            b = np.array(pd.DataFrame(sess.run(prediction_, feed_dict={xs: X_train, ys: y_train})).idxmax(axis=1))
            #
            miss_b = np.array(pd.DataFrame(sess.run(prediction_, feed_dict={xs: X_test, ys: y_test})).idxmax(axis=1))
            miss_b = miss_b.reshape([237605,1])
            print(miss_b.shape, X_test.shape)

            want = np.concatenate((X_test, np.array(pd.DataFrame(y_test).idxmax(axis=1).reshape([237605,1]))), axis=1)
            print(want.shape)
            want[index_data, 56] = miss_b[index_data,0]
            print(want.shape)
            pd.DataFrame(want).to_csv('want.csv')
        if i % 100 == 0:
            print(confusion_matrix(a, b))
basicNN()

"""
('cost:', 6925412.0)
[[195241  39822  18844      0   3153   4702]
 [ 24164  20165  14035      0    500   5138]
 [ 13076   3317  12835      0     62   2356]
 [  3762    592    935      0     53   1187]
 [   349    115     49      0     11     32]
 [  6213   2453   3039      0     52   2919]]


y_train distribution:
261762
64002
31646
6529
556
14676


"""