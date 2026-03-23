import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


# ===================== КОРРЕЛЯЦИИ =====================

def pearson_corr(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x, y):
    return float(spearmanr(x, y).correlation)


def quadrant_corr(x, y):
    mx = np.median(x)
    my = np.median(y)

    sx = np.sign(x - mx)
    sy = np.sign(y - my)

    return float(np.mean(sx * sy))


# ===================== ГЕНЕРАЦИЯ =====================

def bivariate_normal(n, rho, rng):
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    return rng.multivariate_normal(mean, cov, n)


def bivariate_mix(n, rng):
    mean = [0, 0]
    p1 = 0.9

    labels = rng.random(n) < p1
    x = np.empty((n, 2))

    cov1 = [[1, 0.9], [0.9, 1]]
    cov2 = [[100, -90], [-90, 100]]

    x[labels] = rng.multivariate_normal(mean, cov1, labels.sum())
    x[~labels] = rng.multivariate_normal(mean, cov2, (~labels).sum())

    return x


# ===================== ЭЛЛИПС =====================

def ellipse_points(mx, my, sx, sy, rho, C=5.991):
    # 95% уровень (χ² с 2 степенями свободы)

    numerator = 2 * rho * sx * sy
    denominator = sx**2 - sy**2
    alpha = 0.5 * np.arctan2(numerator, denominator)

    term = np.sqrt((sx**2 - sy**2)**2 + 4*(rho*sx*sy)**2)
    l1 = 0.5 * (sx**2 + sy**2 + term)
    l2 = 0.5 * (sx**2 + sy**2 - term)

    a = np.sqrt(C * l1)
    b = np.sqrt(C * l2)

    t = np.linspace(0, 2*np.pi, 400)
    x = a * np.cos(t)
    y = b * np.sin(t)

    xr = x*np.cos(alpha) - y*np.sin(alpha)
    yr = x*np.sin(alpha) + y*np.cos(alpha)

    return mx + xr, my + yr


# ===================== ГРАФИКИ =====================

def plot_normals(sample_size, rhos, rng):
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for i, n in enumerate(sample_size):
        for j, rho in enumerate(rhos):
            sample = bivariate_normal(n, rho, rng)

            x, y = sample[:, 0], sample[:, 1]

            mx, my = np.mean(x), np.mean(y)
            sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)

            rp = pearson_corr(x, y)
            rs = spearman_corr(x, y)
            rq = quadrant_corr(x, y)

            ex, ey = ellipse_points(mx, my, sx, sy, rho)

            ax = axes[i, j]
            ax.scatter(x, y, s=15)
            ax.plot(ex, ey, color='red')
            ax.scatter(mx, my, marker='x', color='black')

            ax.set_title(
                f"n={n}, ρ={rho}\n"
                f"r={rp:.2f}, rs={rs:.2f}, rq={rq:.2f}"
            )

            ax.grid(True)
            ax.axis('equal')

    plt.suptitle("Двумерное нормальное распределение N(0,0,1,1,ρ)")
    plt.tight_layout()


def plot_mix(sample_size, rng):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for j, n in enumerate(sample_size):
        sample = bivariate_mix(n, rng)

        x, y = sample[:, 0], sample[:, 1]

        mx, my = np.mean(x), np.mean(y)
        sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)

        rp = pearson_corr(x, y)
        rs = spearman_corr(x, y)
        rq = quadrant_corr(x, y)

        rho = np.corrcoef(x, y)[0, 1]

        ex, ey = ellipse_points(mx, my, sx, sy, rho)

        ax = axes[j]
        ax.scatter(x, y, s=15)
        ax.plot(ex, ey, color='red')
        ax.scatter(mx, my, marker='x', color='black')

        ax.set_title(
            f"mixture, n={n}\n"
            f"r={rp:.2f}, rs={rs:.2f}, rq={rq:.2f}"
        )

        ax.grid(True)
        ax.axis('equal')

    plt.suptitle("Смесь распределений")
    plt.tight_layout()


# ===================== ОСНОВНОЕ =====================

def main():
    sample_size = [20, 60, 100]
    rhos = [0, 0.5, 0.9]
    n_exp = 1000

    rng = np.random.default_rng(42)

    print("===== Нормальное распределение =====\n")

    for rho in rhos:
        for n in sample_size:
            p, s, q = [], [], []

            for _ in range(n_exp):
                sample = bivariate_normal(n, rho, rng)
                x, y = sample[:, 0], sample[:, 1]

                p.append(pearson_corr(x, y))
                s.append(spearman_corr(x, y))
                q.append(quadrant_corr(x, y))

            print(f"n={n}, rho={rho}")
            print(f"Pearson:  mean={np.mean(p):.4f}, var={np.var(p, ddof=1):.4f}")
            print(f"Spearman: mean={np.mean(s):.4f}, var={np.var(s, ddof=1):.4f}")
            print(f"Quadrant: mean={np.mean(q):.4f}, var={np.var(q, ddof=1):.4f}\n")

    print("===== Смесь =====\n")

    for n in sample_size:
        p, s, q = [], [], []

        for _ in range(n_exp):
            sample = bivariate_mix(n, rng)
            x, y = sample[:, 0], sample[:, 1]

            p.append(pearson_corr(x, y))
            s.append(spearman_corr(x, y))
            q.append(quadrant_corr(x, y))

        print(f"n={n}")
        print(f"Pearson:  mean={np.mean(p):.4f}, var={np.var(p, ddof=1):.4f}")
        print(f"Spearman: mean={np.mean(s):.4f}, var={np.var(s, ddof=1):.4f}")
        print(f"Quadrant: mean={np.mean(q):.4f}, var={np.var(q, ddof=1):.4f}\n")

    plot_normals(sample_size, rhos, rng)
    plot_mix(sample_size, rng)

    plt.show()


if __name__ == "__main__":
    main()