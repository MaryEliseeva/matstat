import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Для воспроизводимости
np.random.seed(42)

# 1. Генерация данных
x = np.arange(-1.8, 2.1, 0.2)
n = len(x)
true_a, true_b = 2, 2
epsilon = np.random.normal(0, 1, n)
y = true_a + true_b * x + epsilon

# 2. Данные с выбросами (исправлено)
y_outliers = y.copy()
y_outliers[0] += 10  # y1 + 10
y_outliers[-1] -= 10  # y20 - 10


# 3. МНК (без изменений)
def mnk_fit(x, y):
    x_mean, y_mean = np.mean(x), np.mean(y)
    xy_mean = np.mean(x * y)
    x2_mean = np.mean(x ** 2)
    b1 = (xy_mean - x_mean * y_mean) / (x2_mean - x_mean ** 2)
    b0 = y_mean - b1 * x_mean
    return b0, b1


# 4. МНМ (исправлено)
def mnm_loss(params, x, y):
    a, b = params
    return np.sum(np.abs(y - a - b * x))


def mnm_fit(x, y):
    res = minimize(mnm_loss, [0, 0], args=(x, y))
    return res.x


# 5. Анализ и вывод
def process_experiment(x, y, title):
    a_mnk, b_mnk = mnk_fit(x, y)
    a_mnm, b_mnm = mnm_fit(x, y)

    print(f"\n--- {title} ---")
    # Шапка таблицы
    print(f"{'Метод':<6} {'a':>6} {'Δa':>6} {'δa,%':>8} {'b':>6} {'Δb':>6} {'δb,%':>8}")
    for name, a_val, b_val in [("МНК", a_mnk, b_mnk), ("МНМ", a_mnm, b_mnm)]:
        delta_a = abs(a_val - true_a)
        delta_b = abs(b_val - true_b)
        error_a = (delta_a / true_a) * 100
        error_b = (delta_b / true_b) * 100
        print(f"{name:<6} {a_val:6.3f} {delta_a:6.3f} {error_a:8.2f} {b_val:6.3f} {delta_b:6.3f} {error_b:8.2f}")

    # Графики
    plt.scatter(x, y, color='gray', label='Данные')
    plt.plot(x, a_mnk + b_mnk * x, '--', label='МНК')
    plt.plot(x, a_mnm + b_mnm * x, '-', label='МНМ')
    plt.plot(x, true_a + true_b * x, 'k-', alpha=0.3, label='Эталон')
    plt.title(title)
    plt.legend()
    plt.show()


# Запуск
process_experiment(x, y, "Без выбросов")
process_experiment(x, y_outliers, "С выбросами")