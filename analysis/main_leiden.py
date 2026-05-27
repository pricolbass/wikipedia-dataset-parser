#!/usr/bin/env python3
"""
Точка входа: запуск Leiden с многоуровневой оценкой качества.

Использование:
    python main_leiden.py                    # прогнать все resolution
    python main_leiden.py --resolutions {1.0}  # только resolution={1.0}
    python main_leiden.py --quick            # только [0.8, 1.0, 1.5]
    python main_leiden.py --skip-hypotheses  # без проверки гипотез
"""
import json
import argparse
from datetime import datetime

import config
from graph_loader import load_graph, load_ground_truth, get_gt_article_to_category
import leiden_runner


def parse_args():
    parser = argparse.ArgumentParser(description="Leiden clustering on Wikipedia graph")
    parser.add_argument("--resolutions", nargs='+', type=float, default=None,
                        help="Список resolution (по умолчанию все из config)")
    parser.add_argument("--quick", action="store_true",
                        help="Быстрый тест: только [0.8, 1.0, 1.5]")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Число итераций (по умолчанию из config)")
    parser.add_argument("--skip-hypotheses", action="store_true",
                        help="Пропустить проверку гипотез")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Кластеризация Leiden с многоуровневой оценкой")

    # Выбор resolution
    if args.quick:
        resolutions = [0.8, 1.0, 1.5]
        print(f"Быстрый режим: {resolutions}")
    elif args.resolutions:
        resolutions = args.resolutions
        print(f"Пользовательский resolution: {resolutions}")
    else:
        resolutions = config.LEIDEN_RESOLUTIONS
        print(f"Полный запуск: {resolutions}")

    n_iterations = args.iterations or config.LEIDEN_N_ITERATIONS
    print(f"Итераций: {n_iterations}")

    t_start = datetime.now()

    """Загрузка данных"""
    G = load_graph(directed=False)
    gt = load_ground_truth()

    # Преобразуем GT в формат {статья: категория}
    gt_map = get_gt_article_to_category(gt)

    # Размеры категорий (для многоуровневой оценки)
    from collections import Counter
    cat_sizes = dict(Counter(gt_map.values()))

    print(f"\nGT-статистика:")
    print(f"  Узлов с GT: {len(gt_map):,}")
    print(f"  Категорий:  {len(cat_sizes):,}")
    print(f"  Coverage:   {len(gt_map) / G.vcount() * 100:.1f}%")

    """Прогон по всем resolution"""
    all_results = []
    for res in resolutions:
        result = leiden_runner.run_full_analysis(G, gt_map, cat_sizes, res, n_iterations)
        all_results.append(result)

    """Таблица"""
    print("СВОДНАЯ ТАБЛИЦА")
    print(f"{'Res':>5} | {'K':>7} | {'NMI':>6} | {'ARI':>6} | {'Cov':>5} | "
          f"{'MedK':>5} | {'Giant':>7} | {'GiantSz':>8}")
    print("-" * 50)
    for r in all_results:
        giant_str = f"{r['pred_stats']['n_giants_ge1000']}"
        giant_sz_str = f"{r['pred_stats']['mean_giant_size']:.0f}" if r['pred_stats']['n_giants_ge1000'] > 0 else "-"
        print(f"{r['resolution']:>5.1f} | {r['num_clusters']:>7} | "
              f"{r['nmi_all']:>6.4f} | {r['ari_all']:>6.4f} | "
              f"{r['coverage'] * 100:>4.1f}% | {r['pred_stats']['median_size']:>5.1f} | "
              f"{giant_str:>7} | {giant_sz_str:>8}")

    """Проверка гипотез"""
    hypotheses = None
    if not args.skip_hypotheses:
        L = G.ecount()
        hypotheses = leiden_runner.check_hypotheses(all_results, L)

    """Сохранение"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = config.LEIDEN_RESULTS_DIR / f"multi_resolution_{timestamp}.json"

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'graph_nodes': G.vcount(),
            'graph_edges': G.ecount(),
            'n_iterations': n_iterations,
            'resolutions': resolutions,
            'gt_stats': {
                'n_articles': len(gt_map),
                'n_categories': len(cat_sizes),
            },
            'results': all_results,
            'hypotheses': hypotheses,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nСохранено: {results_path}")

    """Выдача рекомендаций"""
    print("Рекомендация:")
    best = max(all_results,
               key=lambda r: (r['multilevel'].get('10-20', {}).get('nmi') or 0) +
                             (r['multilevel'].get('10-20', {}).get('ari') or 0))
    print(f"Лучший resolution (по диапазону 10-20): {best['resolution']}")
    print(f"  NMI(10-20) = {best['multilevel']['10-20']['nmi']:.4f}")
    print(f"  ARI(10-20) = {best['multilevel']['10-20']['ari']:.4f}")
    print(f"  Кластеров:   {best['num_clusters']}")

    total_time = (datetime.now() - t_start).total_seconds() / 60
    print(f"\nОбщее время: {total_time:.1f} мин")


if __name__ == '__main__':
    main()