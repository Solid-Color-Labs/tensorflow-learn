import pandas as pd

df = pd.read_csv('/Users/oklesing/Desktop/Tensorflow-Bootcamp-master/00-Crash-Course-Basics/salaries.csv')

# Get column Salary from dataframe
salary = df['Salary']
print(salary)

# Get columns Salary and Name from dataframes
# For multiple columns use array
salary_name = df[['Salary', 'Name']]
print(salary_name)

# Get the max Salary
max_salary = df['Salary'].max()
print(max_salary)

# Describe pandas dataframe
description = df.describe()
print(description)

# Boolean filters, just like numpy
filter_df = df['Salary'] > 60000
print(filter_df)

# Print out numbers from filter
filter_df = df[df['Salary'] > 60000]
print(filter_df)

matrix = df.as_matrix()
print(matrix)