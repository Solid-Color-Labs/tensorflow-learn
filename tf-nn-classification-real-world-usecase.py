import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Generate data
num_employees = 10
random_num_users = np.random.randint(0, 500, num_employees).tolist()
random_num_items = []
max_pct_of_items = 30
employee_levels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
random_num_levels = np.random.randint(1, len(employee_levels) + 1, num_employees).tolist()

# Generate a list of items, with the max item count being 30% or less of the number of users
for random_user in random_num_users:
    max_random_items = int(random_user / 100 * max_pct_of_items)
    random_num_item = np.random.randint(0, max_random_items)
    random_num_items.append(random_num_item)

d = {'num_users': random_num_users, 'num_items': random_num_items, 'emp_lvl': random_num_levels}
df = pd.DataFrame(data=d)

# Normalize columns
cols_to_norm = list(df.columns.values)
cols_to_norm.remove('emp_lvl')
df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Continuous values/features | Feature columns
num_users = tf.feature_column.numeric_column('num_users')
num_items = tf.feature_column.numeric_column('num_items')