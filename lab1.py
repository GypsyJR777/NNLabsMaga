import numpy as np
import numba
import time
import matplotlib.pyplot as plt

# Фиксируем рандом для стабильных результатов
np.random.seed(42)

# Параметры нейронной сети
input_size = 100
hidden_size = 1000
output_size = 10
batch_size = 32

# Инициализация весов
weights_1 = np.random.randn(input_size, hidden_size)
weights_2 = np.random.randn(hidden_size, output_size)

# Функция активации ReLU для numba
@numba.jit(nopython=True)
def relu_numba(x):
    return np.maximum(0, x)

# Функция активации Sigmoid для numba (если понадобится)
@numba.jit(nopython=True)
def sigmoid_numba(x):
    return 1 / (1 + np.exp(-x))

# Функция ошибки (например, MSE)
def loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Генерация данных
def generate_batch(batch_size, input_size):
    return np.random.randn(batch_size, input_size), np.tile(np.arange(10), (batch_size, 1))

# 1. Вручную (без оптимизаций)
def forward_manual(X, w1, w2):
    batch_size = X.shape[0]
    hidden = np.zeros((batch_size, hidden_size))
    for i in range(batch_size):
        for j in range(hidden_size):
            hidden[i, j] = np.dot(X[i], w1[:, j])
    hidden = relu_numba(hidden)
    output = np.zeros((batch_size, output_size))
    for i in range(batch_size):
        for j in range(output_size):
            output[i, j] = np.dot(hidden[i], w2[:, j])
    return output

# 2. Numba оптимизация
@numba.jit(nopython=True)
def forward_numba(X, w1, w2):
    hidden = relu_numba(np.dot(X, w1))  # используем Numba-friendly relu
    output = np.dot(hidden, w2)
    return output

# 3. Используем numpy.dot
def forward_numpy(X, w1, w2):
    hidden = relu_numba(np.dot(X, w1))
    output = np.dot(hidden, w2)
    return output

# Обучение и замеры времени
epochs = 100
errors_manual = []
errors_numba = []
errors_numpy = []

start_manual = time.time()
for epoch in range(epochs):
    X_batch, y_batch = generate_batch(batch_size, input_size)
    output = forward_manual(X_batch, weights_1, weights_2)
    errors_manual.append(loss(y_batch, output))
end_manual = time.time()

start_numba = time.time()
for epoch in range(epochs):
    X_batch, y_batch = generate_batch(batch_size, input_size)
    output = forward_numba(X_batch, weights_1, weights_2)
    errors_numba.append(loss(y_batch, output))
end_numba = time.time()

start_numpy = time.time()
for epoch in range(epochs):
    X_batch, y_batch = generate_batch(batch_size, input_size)
    output = forward_numpy(X_batch, weights_1, weights_2)
    errors_numpy.append(loss(y_batch, output))
end_numpy = time.time()

# График изменения функции ошибки
plt.plot(errors_manual, label='Manual')
plt.plot(errors_numba, label='Numba')
plt.plot(errors_numpy, label='Numpy')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.show()

# Вывод времени
print(f"Время выполнения вручную: {end_manual - start_manual} сек.")
print(f"Время выполнения с Numba: {end_numba - start_numba} сек.")
print(f"Время выполнения с Numpy: {end_numpy - start_numpy} сек.")
