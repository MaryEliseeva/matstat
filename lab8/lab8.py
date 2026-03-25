import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def get_confidence_intervals(data, alpha=0.05):
    n = len(data)
    mean = np.mean(data)

    # Исправленное стандартное отклонение
    s = np.std(data, ddof=1)

    # --- 1. Доверительный интервал для математического ожидания ---
    t_quant = stats.t.ppf(1 - alpha / 2, n - 1)
    margin = t_quant * s / np.sqrt(n)
    ci_mean = (mean - margin, mean + margin)

    # --- 2. Доверительный интервал для sigma ---
    chi_low = stats.chi2.ppf(alpha / 2, n - 1)
    chi_high = stats.chi2.ppf(1 - alpha / 2, n - 1)

    ci_sigma = (
        np.sqrt((n - 1) * s ** 2 / chi_high),
        np.sqrt((n - 1) * s ** 2 / chi_low)
    )

    return ci_mean, ci_sigma


# =========================
# ЗАДАНИЕ 1: Генерация данных
# =========================
np.random.seed(42)

n1, n2 = 20, 100
sample1 = np.random.normal(0, 1, n1)
sample2 = np.random.normal(0, 1, n2)

# =========================
# ЗАДАНИЕ 2: Интервалы
# =========================
print("Доверительные интервалы:\n")

for sample in [sample1, sample2]:
    ci_m, ci_s = get_confidence_intervals(sample)

    print(f"Выборка n = {len(sample)}")
    print(f"  m ∈ ({ci_m[0]:.4f}, {ci_m[1]:.4f})")
    print(f"  σ ∈ ({ci_s[0]:.4f}, {ci_s[1]:.4f})")
    print()

# =========================
# ЗАДАНИЕ 3: F-тест
# =========================
alpha = 0.05

# Исправленные дисперсии
s1_sq = np.var(sample1, ddof=1)
s2_sq = np.var(sample2, ddof=1)

# Формируем F-статистику (большая дисперсия в числителе)
if s1_sq > s2_sq:
    F_obs = s1_sq / s2_sq
    df1, df2 = n1 - 1, n2 - 1
else:
    F_obs = s2_sq / s1_sq
    df1, df2 = n2 - 1, n1 - 1

# Критическое значение
F_crit = stats.f.ppf(1 - alpha, df1, df2)

# Двусторонний p-value
p_value = 2 * min(
    stats.f.cdf(F_obs, df1, df2),
    1 - stats.f.cdf(F_obs, df1, df2)
)

print("F-тест на равенство дисперсий:\n")
print(f"F_наблюдаемое = {F_obs:.4f}")
print(f"F_критическое = {F_crit:.4f}")

if F_obs > F_crit:
    print("Вывод: гипотеза о равенстве дисперсий отвергается")
else:
    print("Вывод: нет оснований отвергать гипотезу о равенстве дисперсий")


def plot_sample(data, title):
    x = np.linspace(-4, 4, 1000)

    plt.figure()

    # Гистограмма
    plt.hist(data, bins=10, density=True)

    # Теоретическая плотность N(0,1)
    plt.plot(x, stats.norm.pdf(x, 0, 1))

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("Плотность")
    plt.grid()
    plt.show()


# Графики для выборок
plot_sample(sample1, "Выборка n=20")
plot_sample(sample2, "Выборка n=100")

# Boxplot (сравнение разброса)
plt.figure()
plt.boxplot([sample1, sample2])
plt.xticks([1, 2], ["n=20", "n=100"])
plt.title("Сравнение выборок (boxplot)")
plt.grid()
plt.show()