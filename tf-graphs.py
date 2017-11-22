import tensorflow as tf

# node 1
n1 = tf.constant(1)
n2 = tf.constant(2)
n3 = n1 + n2

with tf.Session() as sess:
    result = sess.run(n3)
print(result)

# Tensorflow default graph
default = tf.get_default_graph()
print(default)

# New graph
g = tf.Graph()
print(g)

graph_one = tf.get_default_graph()
graph_two = tf.Graph()

# Set graph_two as default graph
# TF graphs  have as default function
with graph_two.as_default():
    print(graph_two is tf.get_default_graph())
