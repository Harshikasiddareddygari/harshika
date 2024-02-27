import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)

X = np.random.uniform(1, 10, 20)
Y = np.random.uniform(1, 10, 20)



classes = np.where(X + Y > 10, 1, 0)


plt.scatter(X, Y, c=classes, cmap=plt.cm.brg)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of Training Data')

# Show the plot
plt.show()
