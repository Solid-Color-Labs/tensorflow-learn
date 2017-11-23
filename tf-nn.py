# neural network that performs a very simple linear fit
# to some 2-D data
#
# Steps
# 1. Build a graph
# 2. Initiate the session
# 3. Feed data in and get output
#
# Graph: wx+b=z
# w = weight
# x = placeholder
# b = bias
####################################

import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt

# Example session graph
# Addition operation
# Multiplication Operation
#######################

# Set seed to get same results every time
np.random.seed(101)
tf.set_random_seed(101)

rand_a = np.random.uniform(0, 100, (5, 5))
rand_b = np.random.uniform(0, 100, (5, 1))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

add_op = a + b
mul_op = a * b
with tf.Session() as sess:
    add_result = sess.run(add_op, feed_dict={a: rand_a, b: rand_b})
    print(add_result)
    print('\n')
    mult_result = sess.run(mul_op, feed_dict={a: rand_a, b: rand_b})
    print(mult_result)

##########################
# Example neural network #
##########################
n_features = 10
n_dense_neurons = 3
x = tf.placeholder(tf.float32, (None, n_features))
W = tf.Variable(tf.random_normal([n_features, n_dense_neurons]))  # weights
b = tf.Variable(tf.ones([n_dense_neurons]))  # bias
xW = tf.matmul(x, W)
z = tf.add(xW, b)
a = tf.nn.sigmoid(z)  # activation function

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    layer_out = sess.run(a, feed_dict={x: np.random.random([1, n_features])})  # result of each neuron

#############################
# Simple regression example #
#############################
# For regression example data is needed
# 10 linearly spaced points between 0 and 10
# Adding/Subtracting a bit of noise a bit of noise -1.5 to 1.5 by 10.
# noise is to prevent perfectly straight line for example purposes.
# linspace is a numpy array of evenly spaced numbers over a specified interval, in this case set to 10.
# I created some noisy data to use in the neural network
x_data = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
y_label = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
# plt.plot(x_data, y_label, '*')
# plt.show()

# y = mx + b
# m = slope
# b = intercept
# random values m and b generated using np.random.rand(2)
# m and b are totally random numbers
# it's up to the neural network with the cost function and optimizer
# to fix m and b to create a nice fitted line in the above plot
# plt.plot(x_data,y_label, '*')
m = tf.Variable(0.44)
b = tf.Variable(0.87)
error = 0

for x, y in zip(x_data, y_label):
    y_hat = m * x + b  # Predicted value

    # cost function predicting error
    # measure how off prediction is
    # y is the true value
    # y_hat is predicted value
    # squaring at the end to punish higher errors
    error += (y - y_hat) ** 2

# optimizer minimizes error
# Gradient decent best figures out how to reduce the cost
# learning_rate defines how fast we're going to decent
# to large of a learning rate, means we might overshoot minimum we're looking for
# to small of a learning rate means it takes forever to do gradient decent.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

# train is optimizer trying to minimize error
train = optimizer.minimize(error)

# Initialize all tf variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    training_steps = 100

    for i in range(training_steps):
        sess.run(train)

    final_slope, final_intercept = sess.run([m,b])

x_test = np.linspace(-1, 11, 10)

# y = mx + b
y_pred_plot = final_slope * x_test + final_intercept

plt.plot(x_test, y_pred_plot, 'r')
plt.plot(x_data, y_label, '*')
plt.show()

