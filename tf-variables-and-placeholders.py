import tensorflow as tf

# Variables example
with tf.Session() as sess:
    my_tensor = tf.random_uniform((4, 4), minval=0, maxval=1)
    my_var = tf.Variable(initial_value=my_tensor)

    # Initialize global tf variables in tf session
    # When creating variable, you need to initialize it in session.
    init = tf.global_variables_initializer()
    sess.run(init)

    # Then run variables after initialization
    result = sess.run(my_var)
print(result)

# Placeholders example
with tf.Session() as sess:
    # (None,5) is common placeholder shape
    # Reason for use of none: none can be filled with
    # the actual number of samples in the data, which
    # if fed in batches you may not know beforehand
    ph = tf.placeholder(tf.float32, shape=(None, 5))
