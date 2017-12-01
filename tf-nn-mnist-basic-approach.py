# Basic approach uses softmax activation function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print('*** Understanding the MNIST data start ***')
print('MNIST training images')
print(mnist.train.images)
print()
print('Number of mnist training examples')
print(mnist.train.num_examples)
print()
print('Number of mnist test examples')
print(mnist.test.num_examples)
print()
print('Number of mnist validation/holding examples')
print(mnist.validation.num_examples)
print()
print('Shape of training images')
print(mnist.train.images.shape)
print('Shape is 55000 images each 784 pixels or 28x28 pixels')
print()

# Plot the dataset
single_image = mnist.train.images[1].reshape(28, 28)
plt.imshow(single_image, cmap='gist_gray')
plt.show()

print('MNIST dataset has already been normalized for you')
print('single_image min: ' + str(single_image.min()))
print('single_image max: ' + str(single_image.max()))
print()
print('*** Understanding the MNIST data end ***')

# Create model
# Steps

# 1. Placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])

# 2. Variables
W = tf.Variable(tf.zeros([784, 10]))  # 784 pixels by 10 possible labels
b = tf.Variable(tf.zeros([10]))

# 3. Create graph operations
y = tf.matmul(x, W) + b

# 4. Loss function
y_true = tf.placeholder(tf.float32, [None, 10])  # 10 possible labels
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))

# 5. Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)

# 6. Create session
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)

        sess.run(train, feed_dict={x: batch_x, y_true: batch_y})

    # 7. Evaluate the model
    # argmax: what index position is y the greatest
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print('Prediction accuracy')
    print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_true:mnist.test.labels}))
