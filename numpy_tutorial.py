import numpy as np

# Vanilla Python List
my_list = [1,2,3]
print(my_list)
print()

# Numpy List
arr = np.array(my_list)
print(arr)
print()

# Range of values - similar to python range function
# Default increment by 1
# 0 to n-1
arange = np.arange(0, 10)
print(arange)
print()

# Range of values - similar to python range function
# Increment by 2
# 0 to n-1
arange1 = np.arange(0, 10, 2)
print(arange1)
print()

# Numpy array of zeros with length of 5
zero_array = np.zeros(5)
print(zero_array)
print()

# Numpy 2d array of zeros
# x = 5 y = 3
# Arg passed in is tuple
zero_2d_array = np.zeros((3,5))
print(zero_2d_array)
print()

# Numpy array of ones with length of 3
one_array = np.ones(3)
print(one_array)
print()

# Numpy 2d array of zeros
# x = 5 y = 3
# Arg passed in is tuple
one_2d_array = np.ones((3, 5))
print(one_2d_array)
print()

# Numpy array of evenly spaced numbers over a specified interval
# Interval is set to 10
numpy_linspace = np.linspace(0, 11, 10)
print(numpy_linspace)
print()

# Random integer in range of 0 to n-1
random_integer = np.random.randint(0,10)
print(random_integer)
print()

# Give a seed to the random functions
# Note if you give the same seed before each random function,
# values will always be the same
np.random.seed(101)

random_seeded_integer = np.random.randint(0, 1000)
print(random_seeded_integer)
print()

# Draw random samples from a normal (Gaussian) distribution.
random_distributed_number = np.random.normal()
print(random_distributed_number)
print()

# Useful python operations
##########################
np.random.seed(101)
arr = np.random.randint(0, 100, 10)

# Max number in numpy list
print(arr.max())

# Min number in numpy list
print(arr.min())

# Average number in numpy list
print(arr.mean())

# Index position of max value in array
print(arr.argmax())

# Index position of min value in array
print(arr.argmin())

# Reshape array into 2 by 5 array
arr = arr.reshape(2, 5)
print(arr)

matrix = np.arange(0, 100).reshape(10,10)
print(matrix)

# Get value of matrix
# matrix[row, column]
num = matrix[0,1]
print(num)
num = matrix[4, 3]
print(num)

# All rows in 1st column of matrix
# Colon indicates all rows
# 0 indicates 1st column
# Both rows and columns are 0 indexed
first_column = matrix[:, 0]
print(first_column)

# Return single row
# Colon indicates all columns
# 5 indicates 6th row
# Both rows and columns are 0 indexed
single_row = matrix[5,:]
print(single_row)

# Columns 0 to n-1
# In this case columns 0 to 2
# Rows 0 to n-1
# In this case rows 0 to 2
chunk = matrix[0:3, 0:3]
print(chunk)

# Numpy masking | Performing boolean operation on matrix
##############

# Where is matrix greater than 50
# Returns back matrix of boolean values
masked_matrix = matrix > 50
print(masked_matrix)

# Get actual values greater than 50, not just masked booleans
matrix = matrix[matrix > 50]
print(matrix)