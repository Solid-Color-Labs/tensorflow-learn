# - Pima Indians Diabetes Dataset
# - Tf.estimator API
# - Categorical and Continuous Features
# - LinearClassifier and DNNClassifier

import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

diabetes = pd.read_csv('pima-indians-diabetes.csv')

# List of columns to normalize
cols_to_norm = list(diabetes.columns.values)
cols_to_norm.remove('Age')  # Don't normalize. Will be converted to categorical column
cols_to_norm.remove('Class')  # Don't normalize. Label column trying to predict
cols_to_norm.remove('Group')  # Cannot normalize strings, because column contains strings

# Normalize columns / Clean the data
diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Continuous values/features | Feature columns
num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')

# Categorical features
# 2 main ways to deal with categorical features: 1. vocabulary list 2. hash bucket
# vocab list params: Column name, and list of possible categories. The only possible
# categories in the column are A, B, C, D
#
# If you know the set of all possible feature values of a column and there are only a few of them, you can use
# categorical_column_with_vocabulary_list. If you don't know the set of possible values in advance you can use
# categorical_column_with_hash_bucket
assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group', ['A', 'B', 'C', 'D'])

# hash_bucket_size is the number of groups you expect in the categorical column.
# hash_bucket_size can be greater than the actual group amount, but not less
# Uncomment below line for hash bucket usage instead of vocab usage
# assigned_group = tf.feature_column.categorical_column_with_hash_bucket('Group', hash_bucket_size=10)

#####
# Converting continuous column to categorical column
# Known as feature engineering. Sometimes allows you to get more out of your column
#####
diabetes['Age'].hist(bins=20)
# plt.show()

# if you look at histogram you can see certain age buckets
# using bucketing system you can convert continuous numeric into categorical column
# Doesn't always help. May make things worse.
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[20, 30, 40, 50, 60, 70, 80])

feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin, bmi, diabetes_pedigree, assigned_group, age_buckets]

# Train Test Split
x_data = diabetes.drop('Class', axis=1)  # Drop Class as it's not a feature. x_data contains all features.
labels = diabetes['Class']
X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.33, random_state=101)

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)

# n_classes is euqal to 2, because we're running binary classification
model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=2)

############
# Training #
############

# Train our model
model.train(input_fn=input_func, steps=1000)

# num_epochs of 1 is the default
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)

# Evaluate our model against test data
results = model.evaluate(eval_input_func)
print('Model evaluation against test data results')
print(results)

###############
# Predictions #
###############

# x is equal to whatever new data you had in. We don't have any new data, because we didn't create a holdout dataset.
# That's why we're passing in test data again.
# We're not passing in a y value, because that's what we're predicting for.
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=10, num_epochs=1, shuffle=False)

# Predictions is going to be a generator, so we can cast to a list
predictions = model.predict(pred_input_func)

my_pred = list(predictions)

# Printing out first 2 results, as to not clutter up output
print()
print('First 2 predictions from model')
print(my_pred[:2])

###################################
# Dense neural network classifier #
###################################

# In DNN, If you define your own categorical column using a vocabulary list, you need to pass it into embedding column.
# Pass in previous categorical column which was assigned_group
# We had 4 groups: 'A','B','C','D'
embedded_group_col = tf.feature_column.embedding_column(assigned_group, dimension=4)

# Reset feature columns, but replace group feature, assigned_group, with embedded_group_col
feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin, bmi, diabetes_pedigree, embedded_group_col,
             age_buckets]

input_func = tf.estimator.inputs.pandas_input_fn(X_train, y_train, batch_size=10, num_epochs=1000, shuffle=True)

# hidden_units defines how many neurons and how many layers
# provide hidden_units a list of neurons per layer
# 3 layers with 10 neurons each.
# It's densely connected. Every neuron is connected to every other neuron.
# More hidden units, the longer it will take to train
dnn_model = tf.estimator.DNNClassifier(hidden_units=[10, 10, 10], feature_columns=feat_cols, n_classes=2)

dnn_model.train(input_func, steps=1000)

# Number of epochs is 1 because it's an evaluation
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
results = dnn_model.evaluate(eval_input_func)
# If you compare regression and dnn results, dnn model is slightly more precise than regression model
print()
print('DNN Model evaluation against test data results')
print(results)
