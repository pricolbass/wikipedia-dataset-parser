#!/usr/bin/env python3
"""
Генерация графиков для Практической части.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Пути
LEIDEN_RESULTS_PATH = Path("./output/leiden/multi_resolution_20260527_023128.json")
OUTPUT_DIR = Path("./output/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300


def load_results():
    with open(LEIDEN_RESULTS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


# График 1: Бимодальность (Сводный)
def plot_cluster_size_bimodality(results):
    """Строит график, сравнивающий количество микро-кластеров и гигантских."""
    print("График 1: Бимодальность (Сводный):")

    resolutions = []
    n_micro = []
    n_giants = []

    for r in results['results']:
        resolutions.append(r['resolution'])
        n_micro.append(r['pred_stats']['n_micro_le5'])
        n_giants.append(r['pred_stats']['n_giants_ge1000'])

    x = np.arange(len(resolutions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width / 2, n_micro, width, label='Микро (≤5)', color='steelblue')
    rects2 = ax.bar(x + width / 2, n_giants, width, label='Гиганты (≥1000)', color='coral')

    ax.set_ylabel('Количество кластеров', fontsize=12)
    ax.set_xlabel('Resolution', fontsize=12)
    ax.set_title('Бимодальность распределения размеров кластеров Leiden', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r}' for r in resolutions])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Подписи значений на столбцах
    for bar in rects1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 5,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    for bar in rects2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 5,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bimodality_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Сохранено: {OUTPUT_DIR / 'bimodality_summary.png'}")

# График 2: Метрики против Resolution
def plot_metrics_vs_resolution(results):
    """Линейный график: NMI и ARI против resolution."""
    print("График 2: Метрики против resolution:")

    resolutions = []
    nmi_all = []
    ari_all = []
    nmi_50plus = []
    ari_50plus = []

    for r in results['results']:
        resolutions.append(r['resolution'])
        nmi_all.append(r['nmi_all'])
        ari_all.append(r['ari_all'])
        nmi_50plus.append(r['multilevel']['50+']['nmi'])
        ari_50plus.append(r['multilevel']['50+']['ari'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # NMI
    ax1.plot(resolutions, nmi_all, 'o-', linewidth=2, markersize=8, label='Общий NMI')
    ax1.plot(resolutions, nmi_50plus, 's-', linewidth=2, markersize=8, label='NMI (50+ узлов)')
    ax1.set_xlabel('Resolution', fontsize=12)
    ax1.set_ylabel('NMI', fontsize=12)
    ax1.set_title('Зависимость NMI от параметра resolution', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # ARI
    ax2.plot(resolutions, ari_all, 'o-', linewidth=2, markersize=8, label='Общий ARI')
    ax2.plot(resolutions, ari_50plus, 's-', linewidth=2, markersize=8, label='ARI (50+ узлов)')
    ax2.set_xlabel('Resolution', fontsize=12)
    ax2.set_ylabel('ARI', fontsize=12)
    ax2.set_title('Зависимость ARI от параметра resolution', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "metrics_vs_resolution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Сохранено: {OUTPUT_DIR / 'metrics_vs_resolution.png'}")


# График 3: Метрики и Размер GT
def plot_metrics_vs_gt_size(results, resolution=1.0):
    """Линейный график: NMI и ARI и размер GT-категории."""
    print(f"График 3: Метрики и размер GT (res={resolution}):")

    res_data = None
    for r in results['results']:
        if abs(r['resolution'] - resolution) < 0.01:
            res_data = r
            break

    if not res_data:
        print(f"Resolution {resolution} не найден")
        return

    ranges = ['3-10', '10-20', '20-50', '50+']
    nmi_values = []
    ari_values = []

    for rng in ranges:
        nmi_values.append(res_data['multilevel'][rng]['nmi'])
        ari_values.append(res_data['multilevel'][rng]['ari'])

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(ranges))
    width = 0.35

    bars1 = ax.bar(x - width / 2, nmi_values, width, label='NMI', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width / 2, ari_values, width, label='ARI', color='coral', alpha=0.8)

    ax.set_xlabel('Размер GT-категории', fontsize=12)
    ax.set_ylabel('Значение метрики', fontsize=12)
    ax.set_title(f'Зависимость метрик от размера GT-категории (resolution={resolution})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(ranges)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0, 0.8)

    # Добавляем значения над барами
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"metrics_vs_gt_size_res{resolution}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Сохранено: {OUTPUT_DIR / f'metrics_vs_gt_size_res{resolution}.png'}")


# График 4: Heatmap NMI
def plot_nmi_heatmap(results):
    """Heatmap: NMI для каждого диапазона GT при разных resolution."""
    print("График 4: Heatmap NMI:")

    resolutions = []
    ranges = ['3-10', '10-20', '20-50', '50+']
    nmi_matrix = []

    for r in results['results']:
        resolutions.append(r['resolution'])
        row = [r['multilevel'][rng]['nmi'] for rng in ranges]
        nmi_matrix.append(row)

    nmi_matrix = np.array(nmi_matrix)

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(nmi_matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(np.arange(len(ranges)))
    ax.set_yticks(np.arange(len(resolutions)))
    ax.set_xticklabels(ranges, fontsize=11)
    ax.set_yticklabels([f'{r:.1f}' for r in resolutions], fontsize=11)

    ax.set_xlabel('Размер GT-категории', fontsize=12)
    ax.set_ylabel('Resolution', fontsize=12)
    ax.set_title('NMI для разных диапазонов GT и значений resolution', fontsize=14)

    for i in range(len(resolutions)):
        for j in range(len(ranges)):
            text = ax.text(j, i, f'{nmi_matrix[i, j]:.3f}',
                           ha="center", va="center", color="black", fontsize=10, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('NMI', fontsize=12)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "nmi_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Сохранено: {OUTPUT_DIR / 'nmi_heatmap.png'}")


def main():
    print("Генерация графиков")
    results = load_results()

    plot_cluster_size_bimodality(results)
    plot_metrics_vs_resolution(results)

    plot_metrics_vs_gt_size(results, resolution=1.0)
    plot_metrics_vs_gt_size(results, resolution=2.0)

    plot_nmi_heatmap(results)

    print("\nВсе графики сгенерированы успешно!")


if __name__ == '__main__':
    main()