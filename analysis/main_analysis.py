#!/usr/bin/env python3
"""
Главный скрипт: запускает полный анализ графа Википедии

Использование:
    python main_analysis.py              # полный анализ
    python main_analysis.py --skip-viz   # без графиков
    python main_analysis.py --skip-topo  # без топологических метрик
"""
import json
import argparse
from datetime import datetime

import config
from graph_loader import load_graph, load_ground_truth
import analysis
import visualize


def main():
    parser = argparse.ArgumentParser(description="Анализ графа Википедии")
    parser.add_argument("--skip-viz", action="store_true", help="Пропустить графики")
    parser.add_argument("--skip-topo", action="store_true", help="Пропустить Monte-Carlo (долго)")
    parser.add_argument("--with-directed", action="store_true", help="Загрузить также ориентированный граф (для in/out степеней)")
    args = parser.parse_args()

    print("Полный анализ графа Википедии")

    t_start = datetime.now()
    report = {
        "timestamp": t_start.isoformat(),
        "config": {
            "edge_path": str(config.EDGE_PATH),
            "gt_path": str(config.GT_PATH),
            "power_law_k_min": config.POWER_LAW_K_MIN,
            "size_ranges": {k: list(v) for k, v in config.SIZE_RANGES.items()},
            "monte_carlo_sample": config.MONTE_CARLO_SAMPLE_SIZE,
        },
    }

    # Загрузка данных
    G = load_graph(directed=False)

    G_dir = None
    if args.with_directed:
        G_dir = load_graph(directed=True)

    gt = load_ground_truth()

    """Анализ"""
    # 1. Базовая статистика
    stats = analysis.basic_statistics(G, config.PERCENTILES, config.THRESHOLDS)
    report["basic_stats"] = stats

    # 2. Топологические метрики (может быть долго)
    if not args.skip_topo:
        topo = analysis.topology_metrics(G)
        report["topology"] = topo

    # 3. Степенной закон
    power_law = analysis.power_law_analysis(G, G_dir)
    report["power_law"] = power_law

    # 4. Анализ GT
    community = analysis.community_analysis(gt)
    report["ground_truth"] = community

    """Визуализация"""
    if not args.skip_viz:
        gamma_und = power_law["undirected"]["gamma"]
        k_min_und = power_law["undirected"]["k_min"]
        degrees = G.degree()
        visualize.plot_all(degrees, stats, gamma_und, k_min_und, gt)

    """Сохранение отчёта"""
    report["total_time_seconds"] = (datetime.now() - t_start).total_seconds()

    report_path = config.RESULTS_DIR / "report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Отчёт сохранён: {report_path}")
    print(f"Общее время: {report['total_time_seconds']:.1f} сек")


if __name__ == "__main__":
    main()