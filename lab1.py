import math
import matplotlib.pyplot as plt

N_STUDENT = 21
A_BOUND = 0.0
B_BOUND = math.pi
PERIOD = B_BOUND - A_BOUND
OMEGA_0 = 2.0 * math.pi / PERIOD

def target_function(x, n=N_STUDENT):
    return n * math.sin(math.pi * n * x)

def integrate_simpson(func, a, b, num_intervals=10000):
    if num_intervals % 2 != 0:
        num_intervals += 1
    h = (b - a) / num_intervals
    total = func(a) + func(b)
    for i in range(1, num_intervals):
        x = a + i * h
        if i % 2 == 0:
            total += 2.0 * func(x)
        else:
            total += 4.0 * func(x)
    return (h / 3.0) * total

def compute_coefficients(N, n=N_STUDENT):
    a = [0.0] * (N + 1)
    b = [0.0] * (N + 1)

    integral_a0 = integrate_simpson(lambda x: target_function(x, n), A_BOUND, B_BOUND)
    a[0] = (2.0 / PERIOD) * integral_a0

    for k in range(1, N + 1):
        integral_ak = integrate_simpson(
            lambda x: target_function(x, n) * math.cos(k * OMEGA_0 * x),
            A_BOUND,
            B_BOUND
        )
        a[k] = (2.0 / PERIOD) * integral_ak

        integral_bk = integrate_simpson(
            lambda x: target_function(x, n) * math.sin(k * OMEGA_0 * x),
            A_BOUND,
            B_BOUND
        )
        b[k] = (2.0 / PERIOD) * integral_bk

    return a, b

def fourier_approximation(x, a, b, N):
    res = a[0] / 2.0
    for k in range(1, N + 1):
        res += a[k] * math.cos(k * OMEGA_0 * x) + b[k] * math.sin(k * OMEGA_0 * x)
    return res

def compute_relative_error(y_true, y_approx):
    sum_diff_sq = 0.0
    sum_true_sq = 0.0
    for yt, ya in zip(y_true, y_approx):
        diff = yt - ya
        sum_diff_sq += diff * diff
        sum_true_sq += yt * yt
    if sum_true_sq == 0.0:
        return 0.0
    return (math.sqrt(sum_diff_sq) / math.sqrt(sum_true_sq)) * 100.0

def save_results_to_file(filename, N, a, b, error):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"N = {N}\n")
        f.write(f"Relative error = {error:.6f}%\n\n")
        f.write("Fourier Coefficients:\n")
        f.write(f"a_0 = {a[0]:.8f}\n")
        for k in range(1, N + 1):
            f.write(f"k = {k:2d}: a_k = {a[k]:12.8f} | b_k = {b[k]:12.8f}\n")

def plot_results(a, b, N, x_vals, y_true, y_approx):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].plot(x_vals, y_true, label="f(x)", color="blue", linewidth=1.2)
    axes[0].plot(x_vals, y_approx, label=f"f_N(x) (N={N})", color="red", linestyle="--", linewidth=1.2)
    axes[0].set_title(f"Наближення функції при N = {N}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].grid(True)
    axes[0].legend()

    for k in range(1, min(N + 1, 6)):
        harmonic_y = [
            a[k] * math.cos(k * OMEGA_0 * x) + b[k] * math.sin(k * OMEGA_0 * x)
            for x in x_vals
        ]
        axes[1].plot(x_vals, harmonic_y, label=f"k = {k}")
    axes[1].set_title("Окремі гармоніки")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("Амплітуда")
    axes[1].grid(True)
    axes[1].legend()

    k_indices = list(range(N + 1))
    width = 0.35
    k_a = [k - width / 2.0 for k in k_indices]
    k_b = [k + width / 2.0 for k in k_indices]

    axes[2].bar(k_a, a, width=width, label="a_k", color="darkblue")
    axes[2].bar(k_b, b, width=width, label="b_k", color="darkred")
    axes[2].set_title("Коефіцієнти Фур'є")
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("Значення")
    axes[2].set_xticks(k_indices)
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.show()

def main():
    N = 10
    a, b = compute_coefficients(N)

    num_points = 2000
    step = (B_BOUND - A_BOUND) / (num_points - 1)
    x_vals = [A_BOUND + i * step for i in range(num_points)]
    
    y_true = [target_function(x) for x in x_vals]
    y_approx = [fourier_approximation(x, a, b, N) for x in x_vals]

    error = compute_relative_error(y_true, y_approx)
    save_results_to_file("fourier_results_var21.txt", N, a, b, error)
    plot_results(a, b, N, x_vals, y_true, y_approx)

if __name__ == "__main__":
    main()