import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Generate data
random_num_users = np.random.randint(0, 500, 10).tolist()
random_num_items = []
max_pct_of_items = 30

# Generate a list of items, with the max item count being 30% or less of the number of users
for random_num_user in random_num_users:
    max_random_user = int(random_num_user / 100 * max_pct_of_items)
    random_num_user = np.random.randint(0, max_random_user)
    random_num_items.append(random_num_user)

d = {'Number of users': random_num_users, 'Random number of items': random_num_items}
df = pd.DataFrame(data=d)