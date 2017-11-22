import tensorflow as tf

hello = tf.constant("Hello ")
world = tf.constant("World")

# With keyword ensures that sessions auto closes
with tf.Session() as sess:
    result = sess.run(hello + world)
print(result)

a = tf.constant(10)
b = tf.constant(20)

with tf.Session() as sess:
    result = sess.run(a + b)
print(result)

const = tf.constant(10)

# 4 by 4 matrix of all tens
fill_mat = tf.fill((4, 4), 10)

# 4 by 4 matrix of all zeros
myzeros = tf.zeros((4, 4))

# 4 by 4 matrix of all ones
myones = tf.ones((4, 4))

# Outputs random values from a normal distribution
# Both mean and standard deviation are optional
myrandn = tf.random_normal((4, 4), mean=0, stddev=1.0)

# Outputs random values from a uniform distribution
myrandu = tf.random_uniform((4, 4), minval=0, maxval=1)

# List of tensorflow operations
my_ops = [const, fill_mat, myzeros, myones, myrandn, myrandu]

with tf.Session() as sess:
    for op in my_ops:
        # op.eval() is an alternative syntax to sess,run(op)
        result = sess.run(op)
        print(result)
        print('\n')

# 2 by 2 matrix
a = tf.constant([[1, 2],
                 [3, 4]])
# 2 by 1 matrix
b = tf.constant([[10], [100]])

result = tf.matmul(a, b)

with tf.Session() as sess:
    # result = sess.run(result)
    # Using other tf syntax for example purposes
    result = result.eval()
print(result)
