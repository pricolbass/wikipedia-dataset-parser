"""
Модуль запуска Leiden с многоуровневой оценкой качества.

Использование:
    from leiden_runner import run_leiden, compute_metrics, run_full_analysis
"""
import numpy as np
import igraph as ig
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from collections import Counter
from datetime import datetime

import config


def run_leiden(G: ig.Graph, resolution: float, n_iterations: int = None) -> ig.VertexClustering:
    """Запускает алгоритм Leiden с заданным resolution.

    Args:
        G: неориентированный граф
        resolution: параметр гранулярности (по умолчанию из config)
        n_iterations: число итераций (по умолчанию из config)

    Returns:
        VertexClustering - разбиение графа на сообщества
    """
    if n_iterations is None:
        n_iterations = config.LEIDEN_N_ITERATIONS

    print(f"Leiden: resolution={resolution}, итераций={n_iterations}")
    t0 = datetime.now()

    partition = G.community_leiden(
        objective_function='modularity',
        resolution=resolution,
        n_iterations=n_iterations
    )

    dt = (datetime.now() - t0).total_seconds()
    print(f"Готово за {dt:.1f} сек, кластеров: {len(partition)}")
    return partition

def compute_metrics(
        G: ig.Graph,
        gt_map: dict,
        cat_sizes: dict,
        partition: ig.VertexClustering
) -> dict:
    """Вычисляет NMI, ARI, coverage и многоуровневые метрики за один проход.

    Args:
        G: граф с атрибутом 'name' у вершин
        gt_map: {статья: категория} (с правилом "первая тема")
        cat_sizes: {категория: размер}
        partition: результат Leiden

    Returns:
        dict со всеми метриками
    """
    names = np.array(G.vs['name'])
    cluster_labels = np.array(partition.membership)
    name_to_idx = {name: i for i, name in enumerate(names)}

    # Собираем согласованные данные (только для узлов с GT)
    gt_indices, true_labels, pred_labels = [], [], []
    for article, category in gt_map.items():
        idx = name_to_idx.get(article)
        if idx is not None:
            gt_indices.append(idx)
            true_labels.append(category)
            pred_labels.append(cluster_labels[idx])

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    n_matched = len(true_labels)

    # Coverage
    coverage = n_matched / len(gt_map) if gt_map else 0

    # Общие метрики
    nmi_all = normalized_mutual_info_score(true_labels, pred_labels) if n_matched > 1 else 0
    ari_all = adjusted_rand_score(true_labels, pred_labels) if n_matched > 1 else 0

    # Многоуровневые метрики (по диапазонам размеров GT-категорий)
    multilevel = {}
    for range_name, (lo, hi) in config.SIZE_RANGES.items():
        if hi == float('inf'):
            mask = np.array([cat_sizes[cat] >= lo for cat in true_labels])
        else:
            mask = np.array([lo <= cat_sizes[cat] < hi for cat in true_labels])

        if mask.sum() > 1:
            nmi = normalized_mutual_info_score(true_labels[mask], pred_labels[mask])
            ari = adjusted_rand_score(true_labels[mask], pred_labels[mask])
            multilevel[range_name] = {
                'nmi': round(float(nmi), 4),
                'ari': round(float(ari), 4),
                'n_nodes': int(mask.sum()),
                'n_categories': int(mask.sum() > 0 and np.unique(true_labels[mask]).size or 0)
            }
        else:
            multilevel[range_name] = {'nmi': None, 'ari': None, 'n_nodes': int(mask.sum()), 'n_categories': 0}

    # Статистика предсказанных кластеров (для проверки теоремы Фортунато)
    cluster_sizes = np.array(list(Counter(cluster_labels).values()))
    micro = int(np.sum(cluster_sizes <= 5))
    giants = cluster_sizes[cluster_sizes >= 1000]

    pred_stats = {
        'num_clusters': int(len(cluster_sizes)),
        'median_size': float(np.median(cluster_sizes)),
        'mode_size': int(Counter(cluster_sizes).most_common(1)[0][0]),
        'min_size': int(cluster_sizes.min()),
        'max_size': int(cluster_sizes.max()),
        'mean_size': float(cluster_sizes.mean()),
        'n_micro_le5': micro,
        'n_giants_ge1000': int(len(giants)),
        'mean_giant_size': float(giants.mean()) if len(giants) > 0 else 0,
        'std_giant_size': float(giants.std()) if len(giants) > 0 else 0,
    }

    cluster_labels_dict = {
        names[i]: int(cluster_labels[i])
        for i in range(len(names))
    }

    return {
        'nmi_all': round(float(nmi_all), 4),
        'ari_all': round(float(ari_all), 4),
        'n_all': n_matched,
        'coverage': round(float(coverage), 4),
        'multilevel': multilevel,
        'pred_stats': pred_stats,
        'cluster_labels': cluster_labels_dict,
        'cluster_sizes': cluster_sizes.tolist()
    }

def run_full_analysis(
        G: ig.Graph,
        gt_map: dict,
        cat_sizes: dict,
        resolution: float,
        n_iterations: int = None
) -> dict:
    """Запускает Leiden и вычисляет все метрики для одного resolution."""
    print(f"Resolution = {resolution}")

    # Кластеризация
    partition = run_leiden(G, resolution, n_iterations)

    # Метрики
    t0 = datetime.now()
    metrics = compute_metrics(G, gt_map, cat_sizes, partition)
    dt = (datetime.now() - t0).total_seconds()
    print(f"Метрики за {dt:.1f} сек")

    # Вывоод
    _print_results(resolution, partition, metrics)

    return {
        'resolution': resolution,
        'n_iterations': n_iterations or config.LEIDEN_N_ITERATIONS,
        'num_clusters': len(partition),
        **metrics
    }


def _print_results(resolution: float, partition: ig.VertexClustering, metrics: dict):
    print(f"\nРезультаты (resolution={resolution}):")
    print(f"  Кластеров:          {len(partition):>10,}")
    print(f"  Медианный размер:   {metrics['pred_stats']['median_size']:>10.1f}")
    print(f"  Микро-кластеров:    {metrics['pred_stats']['n_micro_le5']:>10,}")
    print(f"  Гигантских (≥1000): {metrics['pred_stats']['n_giants_ge1000']:>10,}")
    if metrics['pred_stats']['n_giants_ge1000'] > 0:
        print(
            f"  Средний размер:   {metrics['pred_stats']['mean_giant_size']:>10.0f} ± {metrics['pred_stats']['std_giant_size']:.0f}")
    print(f"  Coverage:           {metrics['coverage'] * 100:>9.1f}%")

    print(f"\nМетрики (на {metrics['n_all']:,} узлах):")
    print(f"  NMI = {metrics['nmi_all']:.4f}")
    print(f"  ARI = {metrics['ari_all']:.4f}")

    print(f"\nМногоуровневая оценка:")
    print(f"  {'Диапазон':<10} {'NMI':>8} {'ARI':>8} {'Узлов':>10}")
    print(f"  {'-' * 40}")
    for rng, m in metrics['multilevel'].items():
        nmi_str = f"{m['nmi']:.4f}" if m['nmi'] is not None else "N/A"
        ari_str = f"{m['ari']:.4f}" if m['ari'] is not None else "N/A"
        print(f"  {rng:<10} {nmi_str:>8} {ari_str:>8} {m['n_nodes']:>10,}")

def check_hypotheses(all_results: list, L: int) -> dict:
    """Проверяет 4 гипотезы из раздела 5.2.3 работы.

    H1: средний размер гигантских кластеров ~ √(2L)
    H2: корреляция Спирмена (размер GT <-> NMI) > 0,5
    H3: метрики на 50+ значимо выше, чем на 3-10
    H4: устойчивость размера кластеров к resolution (±30%)
    """
    import math
    from scipy.stats import spearmanr, mannwhitneyu

    theoretical_threshold = math.sqrt(2 * L)
    print(f"\nПроверка гипотез (теор. порог √(2L) ~ {theoretical_threshold:.0f} рёбер):")

    hypotheses = {}

    # H1: средний размер гигантских кластеров
    giant_means = [r['pred_stats']['mean_giant_size']
                   for r in all_results if r['pred_stats']['n_giants_ge1000'] > 0]
    if giant_means:
        avg_giant = np.mean(giant_means)
        # Переводим в рёбра
        avg_giant_edges = avg_giant * 18 * 0.4  # <k> * доля внутренних
        ratio = avg_giant_edges / theoretical_threshold
        hypotheses['H1'] = {
            'description': 'Размер гигантских кластеров ~ √(2L)',
            'theoretical_edges': round(theoretical_threshold),
            'empirical_edges': round(avg_giant_edges),
            'ratio': round(ratio, 3),
            'passed': bool(0.7 <= ratio <= 1.3)  # ±30%
        }
        print(f"  H1: теор. {theoretical_threshold:.0f} рёбер | эмпирика {avg_giant_edges:.0f} | "
              f"отношение {ratio:.2f} {'+' if hypotheses['H1']['passed'] else '-'}")

    # H2: корреляция Спирмена
    best_result = max(all_results, key=lambda r: r['nmi_all'])
    gt_sizes, nmis = [], []
    for rng, m in best_result['multilevel'].items():
        if m['nmi'] is not None:
            lo, hi = config.SIZE_RANGES[rng]
            midpoint = lo if hi == float('inf') else (lo + hi) / 2
            gt_sizes.append(midpoint)
            nmis.append(m['nmi'])

    if len(gt_sizes) >= 3:
        rho, p_val = spearmanr(gt_sizes, nmis)
        hypotheses['H2'] = {
            'description': 'Корреляция Спирмена (размер GT <-> NMI) > 0,5',
            'rho': round(rho, 3),
            'p_value': float(p_val),
            'passed': bool(rho > 0.5 and p_val < 0.05)
        }
        print(f"  H2: ρ = {rho:.3f}, p = {p_val:.2e} {'+' if hypotheses['H2']['passed'] else '+'}")

    # H3: Mann-Whitney (50+ против 3-10)
    nmis_50 = [r['multilevel']['50+']['nmi'] for r in all_results
               if r['multilevel']['50+']['nmi'] is not None]
    nmis_3_10 = [r['multilevel']['3-10']['nmi'] for r in all_results
                 if r['multilevel']['3-10']['nmi'] is not None]
    if nmis_50 and nmis_3_10:
        try:
            u_stat, p_val = mannwhitneyu(nmis_50, nmis_3_10, alternative='greater')
            hypotheses['H3'] = {
                'description': 'NMI(50+) > NMI(3-10) статистически значимо',
                'mean_50plus': round(np.mean(nmis_50), 4),
                'mean_3_10': round(np.mean(nmis_3_10), 4),
                'p_value': float(p_val),
                'passed': bool(p_val < 0.05)
            }
            print(f"  H3: NMI(50+)={np.mean(nmis_50):.3f} vs NMI(3-10)={np.mean(nmis_3_10):.3f}, "
                  f"p={p_val:.2e} {'+' if hypotheses['H3']['passed'] else '-'}")
        except Exception as e:
            print(f"  H3: не удалось проверить ({e})")

    # H4: устойчивость к resolution
    if giant_means:
        std_ratio = np.std(giant_means) / np.mean(giant_means)
        hypotheses['H4'] = {
            'description': 'Устойчивость размера гигантских кластеров (CV < 30%)',
            'mean_size': round(np.mean(giant_means)),
            'std_size': round(np.std(giant_means)),
            'cv': round(std_ratio, 3),
            'passed': bool(std_ratio < 0.3)
        }
        print(f"  H4: mean={np.mean(giant_means):.0f}, std={np.std(giant_means):.0f}, "
              f"CV={std_ratio:.1%} {'+' if hypotheses['H4']['passed'] else '-'}")

    return hypotheses

if __name__ == '__main__':
    print("Это вспомогательная функция Leiden, его запускать не нужно")