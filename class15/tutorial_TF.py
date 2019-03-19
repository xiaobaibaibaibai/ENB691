# Tensorflow is a open source software library for numerical computation using data flow graphs.
# Deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API.
# TensorFlow provides many functions and classes that allow users to build models from scratch.
# Graphs and Session: Tensorflow separates definition of computations from their execution.
#   - Build and define graph.
#   - Use session to execute operations in the graph.

# Start by import package
import numpy as np
import tensorflow as tf
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Basic example
# a = tf.add(10, 20)
# print (a)
# sess = tf.Session()
# print (sess.run(a))

# # More example
# a = 10
# b = 20
# c = tf.add(a, b)
# d = tf.subtract(b, a)
# e = tf.multiply(c, d)
# sess = tf.Session()
# print (sess.run(e))

# # More example
# a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
# b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
# c = tf.matmul(a, b)
# sess = tf.Session()
# print (sess.run(c))


# Specify CPU or GPU
with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], shape=[3, 2])
    c = tf.matmul(a, b)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print (sess.run(c))

# # tf.Variable
# # Value of tf.constant stored in the graph definition.
# a = tf.Variable(2)
# b = tf.Variable([2, 3])
# c = tf.Variable([[0, 1], [2, 3]])
# d = tf.Variable(tf.zeros([784,10]))
# e = tf.Variable(tf.random_normal(mean=0.0, stddev=0.01, shape=[784,10]))
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
# print (sess.run(e))
# print('###################################')
# sess1 = tf.Session()
# sess1.run(init)
# print (sess1.run(e))

# # tf.placeholder
# # Feed the values to placeholders using a dictionary.
# a = tf.placeholder(tf.float32, shape=[3])
# b = tf.constant([7, 7, 7], tf.float32)
# c = tf.add(a, b)
# sess = tf.Session()
# print (sess.run(c, {a: [1, 2, 3]}))

# # More on tf.placeholder
# a = tf.placeholder(tf.float32, shape=[None, 5])
# b = tf.Variable(tf.ones([5, 3]))
# c = tf.matmul(a, b)
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
# d = np.array(np.arange(15)).reshape(3, 5)
# e = np.array(np.arange(20)).reshape(4, 5)
# print (sess.run(c, {a: d}))
# print (sess.run(c, {a: e}))

# # Build Softmax classifier same as in Homework 2
# from keras.datasets import cifar10
# (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
# xVal = xTrain[49000:, :].astype(np.float)
# yVal = np.squeeze(yTrain[49000:, :])
# xTrain = xTrain[:49000, :].astype(np.float)
# yTrain = np.squeeze(yTrain[:49000, :])
# yTest = np.squeeze(yTest)
# xTest = xTest.astype(np.float)

# # Mean Image
# meanImage = np.mean(xTrain, axis=0)
# xTrain -= meanImage
# xVal -= meanImage
# xTest -= meanImage

# # Reshape data from channel to rows
# xTrain = np.reshape(xTrain, (xTrain.shape[0], -1))
# xVal = np.reshape(xVal, (xVal.shape[0], -1))
# xTest = np.reshape(xTest, (xTest.shape[0], -1))

# # Add bias dimension columns
# xTrain = np.hstack([xTrain, np.ones((xTrain.shape[0], 1))])
# xVal = np.hstack([xVal, np.ones((xVal.shape[0], 1))])
# xTest = np.hstack([xTest, np.ones((xTest.shape[0], 1))])


# # Build graph
# with tf.device('/gpu:0'):
#     x = tf.placeholder(tf.float32, shape=[None, 3073])
#     W = tf.Variable(tf.random_normal(mean=0.0, stddev=0.01, shape=[3073, 10]))
#     y = tf.placeholder(tf.int64, [None])

#     # Calculate score
#     score = tf.matmul(x, W)

#     # Calculate loss
#     meanLoss = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(y, 10), logits=score)) + 5e4*tf.nn.l2_loss(W)

#     # Define optimizer
#     optimizer = tf.train.GradientDescentOptimizer(1e-7)
#     # optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999)
#     trainStep = optimizer.minimize(meanLoss)

#     # Define correct Prediction and accuracy
#     correctPrediction = tf.equal(tf.argmax(score, 1), y)
#     accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))*100

# # Create Session
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# startTime = time.time()
# for i in range(1500):
#     # Mini batch
#     batchID = np.random.choice(xTrain.shape[0], 1000, replace=True)
#     xBatch = xTrain[batchID]
#     yBatch = yTrain[batchID]

#     # Train
#     loss, acc, _ = sess.run([meanLoss, accuracy, trainStep], feed_dict={x: xBatch, y: yBatch})

#     if i % 100 == 0:
#         print('Loop {0} loss {1}'.format(i, loss))

# # Print all accuracy
# print ('Training time: {0}'.format(time.time() - startTime))
# print ('Training acc:   {0}%'.format(sess.run(accuracy, feed_dict={x: xTrain, y: yTrain})))
# print ('Validating acc: {0}%'.format(sess.run(accuracy, feed_dict={x: xVal, y: yVal})))
# print ('Testing acc:    {0}%'.format(sess.run(accuracy, feed_dict={x: xTest, y: yTest})))


# # Exercise: Build 2 layer Neural Network with Tensorflow same as in Homework 3
# def lrelu (tmp):
#   return tf.nn.relu(tmp) - 0.01 * tf.nn.relu(-tmp)
