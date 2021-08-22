from NN.sequential import Sequential
from NN import layers
import numpy as np

x = np.array([1, 2, 3, 4], dtype=float)
model = Sequential()
model.add(layers.Dense(8, activation="relu"))
model.add(layers.Dense(3, activation="relu"))
y = model.call(x)
print(y)
