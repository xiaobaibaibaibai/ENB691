import numpy as np
import tensorflow as tf
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# w = tf.get_variable("w", shape=[64, 102]) # Stride = 2, ((32-7)/2)+1 = 13, 13*13*32=5408
# b = tf.get_variable("b", shape=[2])
# shape = w.get_shape().as_list()

z_dim = 100
y_dim = 2
batch_size = 64

# z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z')
z = tf.get_variable("z", shape=[batch_size, z_dim])

# y = tf.placeholder(tf.float32, [batch_size, y_dim], name='y')
y = tf.get_variable("y", shape=[batch_size, y_dim])

inputs = tf.concat(axis=1, values=[z, y])

shape = inputs.get_shape().as_list()

output_size = 1024
matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32)
net1 = tf.matmul(inputs, matrix)

net2 = tf.get_variable("net2", [64, 16, 16, 128], tf.float32)

#=========conv1

pre_conv_size = 35

inputs = tf.image.resize_bilinear(net2, [pre_conv_size]*2)
k_h = k_w = 4
d_h = d_w = 1
stddev=0.02
padding='VALID'

filter_c1 = tf.get_variable('filter_c1', [4, 4, 128, 64],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))  
conv = tf.nn.conv2d(inputs, filter_c1,
                    strides=[1, 1, 1, 1],
                    padding=padding)
print(conv.get_shape)
biases1 = tf.get_variable('biases1', [64],
                            initializer=tf.constant_initializer(0.0))
g_rc3 = tf.reshape(tf.nn.bias_add(conv, biases1), conv.get_shape())   # (64, 32, 32, 64)


#=========conv2


pre_conv_size = 50 + (4 - 1) # 53

inputs = tf.image.resize_bilinear(g_rc3, [pre_conv_size]*2) # shape=(64, 53, 53, 64), input
c_dim = 3   # output_dim
k_h = k_w = 4
d_h = d_w = 1
stddev=0.02
padding='VALID'

filter_c2 = tf.get_variable('filter_c2', [4, 4, 64, 3],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))  
conv = tf.nn.conv2d(inputs, filter_c2,
                    strides=[1, 1, 1, 1],
                    padding=padding)
biases2 = tf.get_variable('biases2', [3],
                            initializer=tf.constant_initializer(0.0))
out = tf.reshape(tf.nn.bias_add(conv, biases2), conv.get_shape())   # (64, 50, 50, 3)



sess = tf.Session()
sess.run(tf.global_variables_initializer())

# print(sess.run(inputs))

print(out.get_shape)
