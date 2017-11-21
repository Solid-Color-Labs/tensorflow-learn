import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd

data = np.random.randint(0, 100, (10, 2))

scalar_model = MinMaxScaler()

#############################################
# Fit to training data                      #
# Transform to training data and test data  #
#############################################

# Will convert data into floating points
# scalar_model.fit(data)

# Transforms data
# scalar_model.transform(data)

# Fit and transform in 1 step
scalar_model.fit_transform(data)

# ML Supervisied learning model
############
my_data = np.random.randint(0, 101, (50, 4))

# 3 features (f1, f2, f3)
# Predicting the label
df = pd.DataFrame(data=np.random.randint(0, 101, (50, 4)), columns=['f1', 'f2', 'f3', 'label'])

# x is the training set
x = df[['f1', 'f2', 'f3']]

# y is test data
y = df['label']

# Train Test Split
# test_size is percentage of data to go to the test set
# random_state is for repeatability. random_state is same as using a seed.
# x_train is feature data for the training set.
# x_test evaluate model using test data. Tensorflow will predict what labels should be.
# For true evaluation you can then compare predicted values against y_test values.
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=101)
print(x_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_train.shape)
