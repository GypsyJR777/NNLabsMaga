import numpy as np
import numba
import time
import matplotlib.pyplot as plt

np.random.seed(42)
input_size = 100
hidden_size = 1000
output_size = 10
batch_size = 32
learning_rate = 0.001

weights_1 = np.random.randn(input_size, hidden_size) * 0.01
weights_2 = np.random.randn(hidden_size, output_size) * 0.01

# Функция активации ReLU
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Генерация данных
def generate_batch(batch_size, input_size, output_size):
    X_batch = np.random.randn(batch_size, input_size)
    y_batch = np.tile(np.arange(output_size), (batch_size, 1))
    return X_batch, y_batch

# 1. Прямой и обратный проход вручную без dot
def forward_backward_manual(X, y, w1, w2):
    batch_size = X.shape[0]

    # Прямой проход
    hidden = np.zeros((batch_size, hidden_size))
    for i in range(batch_size):
        for j in range(hidden_size):
            for k in range(input_size):
                hidden[i, j] += X[i, k] * w1[k, j]
            hidden[i, j] = max(0, hidden[i, j])

    output = np.zeros((batch_size, output_size))
    for i in range(batch_size):
        for j in range(output_size):
            for k in range(hidden_size):
                output[i, j] += hidden[i, k] * w2[k, j]

    error = output - y

    # Обратное
    hidden_error = np.zeros((batch_size, hidden_size))
    for i in range(batch_size):
        for j in range(hidden_size):
            hidden_error[i, j] = np.sum(error[i] * w2[j, :]) * (1 if hidden[i, j] > 0 else 0)

    grad_w2 = np.zeros_like(w2)
    for i in range(hidden_size):
        for j in range(output_size):
            grad_w2[i, j] = np.mean(hidden[:, i] * error[:, j])
    
    grad_w1 = np.zeros_like(w1)
    for i in range(input_size):
        for j in range(hidden_size):
            grad_w1[i, j] = np.mean(X[:, i] * hidden_error[:, j])

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

    return output, np.mean(error ** 2)

# 2. Прямой и обратный проход с Numba
@numba.jit(nopython=True)
def forward_backward_numba(X, y, w1, w2, lr):
    batch_size = X.shape[0]
    
    hidden = np.zeros((batch_size, hidden_size))
    for i in range(batch_size):
        for j in range(hidden_size):
            for k in range(input_size):
                hidden[i, j] += X[i, k] * w1[k, j]
            hidden[i, j] = max(0, hidden[i, j])

    output = np.zeros((batch_size, output_size))
    for i in range(batch_size):
        for j in range(output_size):
            for k in range(hidden_size):
                output[i, j] += hidden[i, k] * w2[k, j]

    error = output - y
    hidden_error = np.zeros((batch_size, hidden_size))
    for i in range(batch_size):
        for j in range(hidden_size):
            hidden_error[i, j] = np.sum(error[i] * w2[j, :]) * (1 if hidden[i, j] > 0 else 0)

    grad_w2 = np.zeros_like(w2)
    for i in range(hidden_size):
        for j in range(output_size):
            grad_w2[i, j] = np.mean(hidden[:, i] * error[:, j])

    grad_w1 = np.zeros_like(w1)
    for i in range(input_size):
        for j in range(hidden_size):
            grad_w1[i, j] = np.mean(X[:, i] * hidden_error[:, j])

    w1 -= lr * grad_w1
    w2 -= lr * grad_w2

    return output, np.mean(error ** 2)

# 3. Прямой и обратный проход с использованием numpy.dot
def forward_backward_numpy(X, y, w1, w2):
    hidden = relu(np.dot(X, w1))
    output = np.dot(hidden, w2)

    error = output - y

    hidden_error = np.dot(error, w2.T) * relu_derivative(hidden)
    grad_w2 = np.dot(hidden.T, error) / batch_size
    grad_w1 = np.dot(X.T, hidden_error) / batch_size

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

    return output, np.mean(error ** 2)


# Обучение и замер времени
epochs = 100
errors_manual = []
errors_numba = []
errors_numpy = []

start_manual = time.time()
for epoch in range(epochs):
    X_batch, y_batch = generate_batch(batch_size, input_size, output_size)
    output, error = forward_backward_manual(X_batch, y_batch, weights_1, weights_2)
    errors_manual.append(error)
end_manual = time.time()

start_numba = time.time()
for epoch in range(epochs):
    X_batch, y_batch = generate_batch(batch_size, input_size, output_size)
    output, error = forward_backward_numba(X_batch, y_batch, weights_1, weights_2, learning_rate)
    errors_numba.append(error)
end_numba = time.time()

start_numpy = time.time()
for epoch in range(epochs):
    X_batch, y_batch = generate_batch(batch_size, input_size, output_size)
    output, error = forward_backward_numpy(X_batch, y_batch, weights_1, weights_2)
    errors_numpy.append(error)
end_numpy = time.time()

# Построение графика функции ошибки
plt.plot(errors_manual, label='Manual')
plt.plot(errors_numba, label='Numba')
plt.plot(errors_numpy, label='NumPy')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.show()

# Сравнение времени выполнения
print(f"Время выполнения вручную: {end_manual - start_manual} секунд")
print(f"Время выполнения с Numba: {end_numba - start_numba} секунд")
print(f"Время выполнения с NumPy: {end_numpy - start_numpy} секунд")

# Оценка тактов на умножение (примерная для одной операции)
cpu_clock_speed = 3.5e9  # Примерная частота процессора в Гц
manual_time_per_epoch = (end_manual - start_manual) / epochs
num_operations = batch_size * (input_size * hidden_size + hidden_size * output_size)
clocks_per_mul_manual = (manual_time_per_epoch * cpu_clock_speed) / num_operations
print(f"Среднее количество тактов на 1 умножение (ручной способ): {clocks_per_mul_manual:.2f}")
