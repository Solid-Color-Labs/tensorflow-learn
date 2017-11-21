import numpy as np
import pandas as pd
import matplotlib.pylab as plt

# use plt.show() to display plot for matplotlib

x = np.arange(0,10)
y = x**2

# Basic graph
# plt.plot(x,y)
# plt.show()

# Basic graph with marker
# plt.plot(x,y,'*')
# plt.show()

# Basic graph with color
# plt.plot(x,y,'red')
# plt.show()

# Basic graph with red dashed line
# plt.plot(x,y,'r--')
# plt.show()

# # Limit x axis axis on graph
# plt.plot(x,y,'r--')
# plt.xlim(0,4)
# # Limit x axis axis on graph
# plt.ylim(0,10)
# # Add title to plot
# plt.title('TITLE')
# # Add x label
# plt.xlabel('X LABEL')
# #Add y label
# plt.ylabel('Y LABEL')
# plt.show()

# Plot with default color
# matrix = np.arange(0,100).reshape(10,10)
# plt.imshow(matrix)
# plt.show()

# Plot with coolwarm color
# matrix = np.arange(0,100).reshape(10,10)
# plt.imshow(matrix,cmap='coolwarm')
# plt.show()

# Random plot with colorbar
# matrix = np.random.randint(0,1000,(10,10))
# plt.imshow(matrix)
# plt.colorbar()
# plt.show()

# Pandas datavisualization, builds on top of matplotlib.
# Adds nice api for dataframe.
df = pd.read_csv('/Users/oklesing/Desktop/Tensorflow-Bootcamp-master/00-Crash-Course-Basics/salaries.csv')
df.plot(x='Salary',y='Age',kind='scatter')
plt.show()