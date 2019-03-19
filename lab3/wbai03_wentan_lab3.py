import numpy as np
import tensorflow as tf
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.datasets import cifar10

def lrelu (tmp):
  return tf.nn.relu(tmp) - 0.01 * tf.nn.relu(-tmp)

(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
xVal = xTrain[49000:, :].astype(np.float)
yVal = np.squeeze(yTrain[49000:, :])
xTrain = xTrain[:49000, :].astype(np.float)
yTrain = np.squeeze(yTrain[:49000, :])
yTest = np.squeeze(yTest)
xTest = xTest.astype(np.float)

# Mean Image
meanImage = np.mean(xTrain, axis=0)
xTrain -= meanImage
xVal -= meanImage
xTest -= meanImage

# Reshape data from channel to rows
xTrain = np.reshape(xTrain, (xTrain.shape[0], -1))
xVal = np.reshape(xVal, (xVal.shape[0], -1))
xTest = np.reshape(xTest, (xTest.shape[0], -1))

# Add bias dimension columns
xTrain = np.hstack([xTrain, np.ones((xTrain.shape[0], 1))])
xVal = np.hstack([xVal, np.ones((xVal.shape[0], 1))])
xTest = np.hstack([xTest, np.ones((xTest.shape[0], 1))])

params = dict()
params['w1'] = None
params['b1'] = None
params['w2'] = None
params['b2'] = None

with tf.device('/cpu:0'):
    x = tf.placeholder(tf.float32, shape=[None, 3073])
    y = tf.placeholder(tf.int64, [None])

    hiddenNeurons = 100

    inputDim = xTrain.shape[1]
    hiddenDim = hiddenNeurons
    outputDim = 10

    sd = 0.0001
    params['w1'] = tf.Variable(tf.random_normal(mean=0.0, stddev=sd, shape=[inputDim, hiddenDim]))
    params['b1'] = tf.Variable(tf.random_normal(mean=0.0, stddev=sd, shape=[hiddenDim]))
    params['w2'] = tf.Variable(tf.random_normal(mean=0.0, stddev=sd, shape=[hiddenDim, outputDim]))
    params['b2'] = tf.Variable(tf.random_normal(mean=0.0, stddev=sd, shape=[outputDim]))

    H = tf.matmul(x, params['w1'])
    Hin = tf.add(H, params['b1'])
    Hout = lrelu(Hin)
    tsc = tf.matmul(Hout, params['w2'])
    score = tf.add(tsc, params['b2'])
    score = lrelu(score)

    meanLoss = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(y, 10), logits=score)) + 5e-3*(tf.nn.l2_loss(params['w1']) + tf.nn.l2_loss(params['w2']))

    optimizer = tf.train.GradientDescentOptimizer(5e-3)

    trainStep = optimizer.minimize(meanLoss)

    correctPrediction = tf.equal(tf.argmax(score, 1), y)

    accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))*100

sess = tf.Session()
sess.run(tf.global_variables_initializer())
startTime = time.time()


for i in range(1500):
    # Mini batch
    batchID = np.random.choice(xTrain.shape[0], 1000, replace=True)
    xBatch = xTrain[batchID] # xBatch is : (1000, 3073)
    yBatch = yTrain[batchID] # yBatch is : (1000,)

    loss, acc, _ = sess.run([meanLoss, accuracy, trainStep], feed_dict={x: xBatch, y: yBatch})

    if i % 100 == 0:
        print('Loop {0} loss {1}'.format(i, loss))

# Print all accuracy
print ('Training time: {0}'.format(time.time() - startTime))
print ('Training acc:   {0}%'.format(sess.run(accuracy, feed_dict={x: xTrain, y: yTrain})))
print ('Validating acc: {0}%'.format(sess.run(accuracy, feed_dict={x: xVal, y: yVal})))
print ('Testing acc:    {0}%'.format(sess.run(accuracy, feed_dict={x: xTest, y: yTest})))

