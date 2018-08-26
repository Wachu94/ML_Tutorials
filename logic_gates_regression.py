import numpy as np
from numpy import linalg

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]

for i in range(len(X)):
    X[i].append(1)

Z = linalg.inv(np.matmul(np.transpose(X),X))
alfa = np.matmul(np.matmul(Z, np.transpose(X)), Y)
print(alfa, end="\n\n")

prediction = np.matmul(X, alfa)

print(prediction, end="\n\n")

# for i in range(len(prediction)):
#     print(round(prediction[i][0]))
#     print(np.argmax(prediction[i]))