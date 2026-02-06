# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize the parameters (slope and intercept) and set the learning rate.

2. Calculate the predicted profit using the linear regression equation.

3. Compute the error and update the parameters using gradient descent.

4. Repeat the process until the model converges and predict the profit for a given city population.

## Program:
```
# Linear Regression using Gradient Descent

import numpy as np
import matplotlib.pyplot as plt

# Step 1: Training data (Population vs Profit)
X = np.array([6.11, 5.27, 8.52, 7.92, 4.45, 6.68])
y = np.array([17.59, 9.13, 13.66, 11.85, 5.20, 12.05])

# Step 2: Normalize input (optional but good practice)
X = (X - np.mean(X)) / np.std(X)

# Step 3: Initialize parameters
m = 0      # slope
c = 0      # intercept
alpha = 0.01   # learning rate
epochs = 1000
n = len(X)

# Step 4: Gradient Descent
for i in range(epochs):
    y_pred = m * X + c
    dm = (-2/n) * sum(X * (y - y_pred))
    dc = (-2/n) * sum(y - y_pred)
    m = m - alpha * dm
    c = c - alpha * dc

# Step 5: Prediction
population = float(input("Enter city population: "))
population = (population - np.mean(X)) / np.std(X)
profit = m * population + c

print("Predicted Profit:", round(profit, 2))

# Step 6: Visualization
plt.scatter(X, y, color='blue')
plt.plot(X, m*X + c, color='red')
plt.xlabel("Population")
plt.ylabel("Profit")
plt.title("Linear Regression using Gradient Descent")
plt.show()

```

## Output:
<img width="563" height="455" alt="image" src="https://github.com/user-attachments/assets/6b85c3ad-a3ec-44e0-872b-08024a6169a4" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
