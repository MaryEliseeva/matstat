import numpy as np
import scipy.stats as stats

def chi2_goodness_of_fit(sample, alpha=0.05, label="Выборка"):
    n = len(sample)

    # 1. Оценка параметров по ММП
    mu = np.mean(sample)
    sigma = np.std(sample, ddof=0)

    # 2. Определяем число интервалов
    if n >= 50:
        k = int(np.ceil(1 + 3.3 * np.log10(n)))  # формула Старджесса
    else:
        k = 10

    # 3. Формируем границы интервалов
    eps = 1e-9
    min_val, max_val = sample.min(), sample.max()
    bins = np.linspace(min_val - eps, max_val + eps, k + 1)
    bins[0] = -np.inf
    bins[-1] = np.inf

    # 4. Наблюдаемые частоты
    observed, _ = np.histogram(sample, bins=bins)
    probs = np.diff(stats.norm.cdf(bins, loc=mu, scale=sigma))
    expected = n * probs

    # 5. Объединяем интервалы с n*p_i < 5
    obs_list = observed.tolist()
    exp_list = expected.tolist()
    bins_list = bins.tolist()
    prob_list = probs.tolist()

    while np.any(np.array(exp_list) < 5) and len(obs_list) > 2:
        idx = next(i for i, e in enumerate(exp_list) if e < 5)
        if idx == 0:
            obs_list[0] += obs_list[1]
            exp_list[0] += exp_list[1]
            prob_list[0] += prob_list[1]
            del obs_list[1]; del exp_list[1]; del bins_list[1]; del prob_list[1]
        else:
            obs_list[idx - 1] += obs_list[idx]
            exp_list[idx - 1] += exp_list[idx]
            prob_list[idx - 1] += prob_list[idx]
            del obs_list[idx]; del exp_list[idx]; del bins_list[idx]; del prob_list[idx]

    k_merged = len(obs_list)
    if k_merged < 3:
        print(f"\n--- {label} (n={n}) ---")
        print("Ошибка: после объединения интервалов осталось менее 3, критерий неприменим.")
        return

    observed = np.array(obs_list)
    expected = np.array(exp_list)
    probs = np.array(prob_list)
    bins = np.array(bins_list)
    df = k_merged - 3  # df = число интервалов - число оцениваемых параметров - 1

    # 6. Вычисляем статистику χ²
    chi2_stat = np.sum((observed - expected) ** 2 / expected)
    critical = stats.chi2.ppf(1 - alpha, df)
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)
    decision = "принимается" if chi2_stat < critical else "отвергается"

    # 7. Вывод результатов
    print(f"\n--- {label} (n={n}) ---")
    print(f"Оценки параметров: μ = {mu:.4f}, σ = {sigma:.4f}")
    print(f"Число интервалов после объединения: {k_merged}")
    print("\nТаблица вычислений χ²:")
    print(f"{'№':<3} {'Интервал':<25} {'n_i':>6} {'p_i':>8} {'n p_i':>8} {'n_i-n p_i':>10} {'(n_i-n p_i)²/(n p_i)':>20}")

    for i in range(k_merged):
        left = bins[i] if i > 0 else -np.inf
        right = bins[i + 1] if i + 1 < len(bins) else np.inf
        interval_str = f"[{-np.inf if np.isinf(left) else left:.2f}, {np.inf if np.isinf(right) else right:.2f}]"
        diff = observed[i] - expected[i]
        contrib = diff ** 2 / expected[i]
        print(f"{i + 1:<3} {interval_str:<25} {observed[i]:6.0f} {probs[i]:8.4f} {expected[i]:8.2f} {diff:10.2f} {contrib:20.4f}")

    print(f"\nχ² наблюдаемое = {chi2_stat:.4f}")
    print(f"χ² критическое = {critical:.4f} (df={df})")
    print(f"p-value = {p_value:.4f}")
    print(f"Гипотеза H0 {decision}")


# --- Основная программа ---
np.random.seed(42)

# 1–3. Нормальное распределение, n=100
sample_norm100 = np.random.normal(0, 1, 100)
chi2_goodness_of_fit(sample_norm100, label="Нормальное N(0,1)")

# 4. Исследование чувствительности, n=20
sample_unif = np.random.uniform(-np.sqrt(3), np.sqrt(3), 20)
chi2_goodness_of_fit(sample_unif, label="Равномерное U(-√3,√3)")

sample_laplace = np.random.laplace(0, 1 / np.sqrt(2), 20)
chi2_goodness_of_fit(sample_laplace, label="Лапласа L(0,1/√2)")

# --- Чувствительность на больших выборках (n=100) ---
sample_unif100 = np.random.uniform(-np.sqrt(3), np.sqrt(3), 100)
chi2_goodness_of_fit(sample_unif100, label="Равномерное U(-√3,√3), n=100")

sample_laplace100 = np.random.laplace(0, 1 / np.sqrt(2), 100)
chi2_goodness_of_fit(sample_laplace100, label="Лапласа L(0,1/√2), n=100")