import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

x_data = np.linspace(0.0, 10, 1000000)
noise = np.random.randn(len(x_data))

# y = mx + b
# b = 5
# Adding noise for example purposes, to kake it purposefully hard.
y_true = (0.5 * x_data) + 5 + noise
x_df = pd.DataFrame(data=x_data, columns=['X Data'])
y_df = pd.DataFrame(data=y_true, columns=['Y'])
my_data = pd.concat([x_df, y_df], axis=1)

# taking a random sample of 250 because 1 million is too much to plot
my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')
# plt.show()

# Create batches of data for training. You can't just feed in all of the data at once.
# In this case we're feeding in 8 at a time.
# Depending on how large the dataset is, you would increase batch size, otherwise it would take forever.
batch_size = 8

# Vars generated via np.random.randn(2)
m = tf.Variable(0.81)
b = tf.Variable(0.17)

xph = tf.placeholder(tf.float32, [batch_size])
yph = tf.placeholder(tf.float32, [batch_size])  # true value
y_model = m * xph + b  # prediction
error = tf.reduce_sum(tf.square(yph - y_model))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # batch_size times batches equals number of samples
    batches = 10000

    # feed in 1000 batches. For each batch there are 8 data points (batch size)
    # We're training model on 8000 data points
    for i in range(batches):
        # ensures we get batch_size number of random data points.
        rand_ind = np.random.randint(len(x_data), size=batch_size)

        feed = {xph: x_data[rand_ind], yph: y_true[rand_ind]}

        sess.run(train, feed_dict=feed)

    # model_b should be close to 5. We set bias to 5, but then also added noise.
    # The model should have reduced the noise, and thus it should be close to 5.
    model_m, model_b = sess.run([m, b])
    y_hat = x_data * model_m + model_b
    my_data.sample(250).plot(kind='scatter', x='X Data', y='Y')
    plt.plot(x_data, y_hat, 'r')
    # plt.show()

################
# TF Estimator #
################
feat_cols = [tf.feature_column.numeric_column('x', shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

# x_train equals 70,000, x_test equals 30,000 due to split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_true, test_size=0.3, random_state=101)

# 'x' matches feature column defined above
input_func = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train, batch_size=8, num_epochs=None, shuffle=True)

# Reason shuffle equals False: Using train input function against for evaluation against a test input function
train_input_func = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train, batch_size=8, num_epochs=1000,
                                                      shuffle=False)

test_input_func = tf.estimator.inputs.numpy_input_fn({'x': x_test}, y_test, batch_size=8, num_epochs=1000,
                                                     shuffle=False)

# Reason for defining number of steps is because we set epochs to None in out input_func.
estimator.train(input_fn=input_func, steps=1000)

# Get metrics on training data
train_metrics = estimator.evaluate(input_fn=train_input_func, steps=1000)

# Get metrics on test data
test_metrics = estimator.evaluate(input_fn=test_input_func, steps=1000)
print('TRAINING DATA METRICS')
print(train_metrics)
# Compare training data metrics to test data metrics
# Good sign of not overfitting is that training data loss is about the same as test data loss
# Should expect test data loss to perform slightly worse than training data
print('TEST DATA METRICS')
print(test_metrics)

# Predict new values
# Given some x value what is it's corrosponding y label according to your model?
brand_new_data = np.linspace(0, 10, 10)
input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x': brand_new_data}, shuffle=False)

# estimator.predict() can be cast to a list, as per below comment
# predictions = list(estimator.predict(input_fn=input_fn_predict))
# print(predictions)

# Plot out predictions
predictions = []
for pred in estimator.predict(input_fn=input_fn_predict):
    predictions.append(pred['predictions'])

my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')

# Points predicted according to model
plt.plot(brand_new_data, predictions, 'r*') # Can also remove * to get line, instead of points
plt.show()
