"""
Модуль визуализации: все графики для работы.
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import config


def plot_degree_distribution_linear(degrees, stats: dict, filename: str = None):
    """Гистограмма степеней (линейная шкала, до k=50) - Рисунок 5.1."""
    if filename is None:
        filename = config.FIGURES_DIR / "degree_distribution_linear.png"

    degrees = np.array(degrees)
    small = degrees[degrees <= 50]

    plt.figure(figsize=(12, 6))
    plt.hist(small, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(stats['mean_degree'], color='red', linestyle='--', linewidth=2,
                label=f'Среднее = {stats["mean_degree"]:.1f}')
    plt.axvline(stats['median_degree'], color='green', linestyle='--', linewidth=2,
                label=f'Медиана = {stats["median_degree"]}')
    plt.xlabel('Степень узла (число гиперссылок)', fontsize=12)
    plt.ylabel('Количество узлов', fontsize=12)
    plt.title('Распределение степеней узлов (до k=50)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"Сохранено: {filename}")


def plot_degree_cdf(degrees, filename: str = None):
    """CDF степеней - Рисунок 5.2."""
    if filename is None:
        filename = config.FIGURES_DIR / "degree_cdf.png"

    degrees = np.array(degrees)
    sorted_deg = np.sort(degrees)
    cdf = np.arange(1, len(sorted_deg) + 1) / len(sorted_deg)

    plt.figure(figsize=(12, 6))
    plt.plot(sorted_deg, cdf, linewidth=2, color='steelblue')
    plt.axhline(0.8, color='red', linestyle='--', alpha=0.7, label='80% узлов')
    plt.axhline(0.5, color='green', linestyle='--', alpha=0.7, label='50% (медиана)')
    plt.axhline(0.95, color='orange', linestyle='--', alpha=0.7, label='95% узлов')
    plt.xscale('log')
    plt.xlabel('Степень узла (log)', fontsize=12)
    plt.ylabel('Доля узлов с degree ≤ k', fontsize=12)
    plt.title('Кумулятивная функция распределения (CDF)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"Сохранено: {filename}")


def plot_power_law(degrees, gamma: float, k_min: int, filename: str = None):
    """Распределение степеней в лог-лог масштабе - Рисунок 5.3."""
    if filename is None:
        filename = config.FIGURES_DIR / "degree_power_law.png"

    degrees = np.array(degrees)
    counts = Counter(degrees)
    k_vals = np.array(sorted(counts.keys()))
    p_k = np.array([counts[k] for k in k_vals]) / len(degrees)

    # Теоретическая кривая
    k_fit = np.linspace(k_min, max(degrees), 100)
    # Нормировка: P(k_min) должно совпадать с эмпирическим
    idx_kmin = np.argmin(np.abs(k_vals - k_min))
    norm_const = p_k[idx_kmin] * (k_min ** gamma)
    p_fit = norm_const * (k_fit ** -gamma)

    plt.figure(figsize=(12, 6))
    plt.scatter(k_vals, p_k, s=15, alpha=0.6, color='steelblue', label='Эмпирическое')
    plt.plot(k_fit, p_fit, 'r--', linewidth=2, label=f'γ = {gamma:.2f}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Степень узла k (log)', fontsize=12)
    plt.ylabel('P(k) (log)', fontsize=12)
    plt.title('Степенной закон распределения степеней', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(filename, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"Сохранено: {filename}")


def plot_community_sizes(gt: dict, filename: str = None):
    """Распределение размеров GT-категорий (до 100 узлов)."""
    if filename is None:
        filename = config.FIGURES_DIR / "community_size_distribution.png"

    sizes = [len(nodes) for nodes in gt.values()]
    sizes_filtered = [s for s in sizes if s <= 100]

    plt.figure(figsize=(10, 6))
    plt.hist(sizes_filtered, bins=50, edgecolor='black', alpha=0.7, color='coral')
    plt.axvline(10, color='red', linestyle='--', linewidth=2, label='Граница 10')
    plt.axvline(20, color='darkred', linestyle='--', linewidth=2, label='Граница 20')
    plt.axvline(50, color='purple', linestyle='--', linewidth=2, label='Граница 50')
    plt.xlabel('Размер сообщества', fontsize=12)
    plt.ylabel('Количество категорий', fontsize=12)
    plt.title('Распределение размеров GT-категорий (до 100)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"Сохранено: {filename}")


def plot_all(degrees, stats: dict, gamma: float, k_min: int, gt: dict):
    """Генерирует все графики сразу."""
    print("Генерация графиков")

    plot_degree_distribution_linear(degrees, stats)
    plot_degree_cdf(degrees)
    plot_power_law(degrees, gamma, k_min)
    plot_community_sizes(gt)

    print("\nВсе графики готовы!")

if __name__ == "__main__":
    print("Это вспомогательная функция визуализации результатов!")