"""
Модуль анализа графа и GT-разметки.

Все функции возвращают dict с результатами (удобно для JSON-отчёта).
"""
import numpy as np
import random
import time
import igraph as ig
from collections import Counter

import config


def basic_statistics(G: ig.Graph, PERCENTILES ,THRESHOLDS) -> dict:
    """Базовая статистика графа: N, L, степени, процентили."""
    print("Базовая статистика графа")

    degrees = np.array(G.degree())
    n_nodes = G.vcount()
    n_edges = G.ecount()

    stats = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "mean_degree": float(np.mean(degrees)),
        "median_degree": int(np.median(degrees)),
        "mode_degree": int(Counter(degrees).most_common(1)[0][0]),
        "max_degree": int(np.max(degrees)),
    }

    # Процентили
    percentiles = {p: int(np.percentile(degrees, p)) for p in PERCENTILES}
    stats["percentiles"] = percentiles

    # Доли узлов с разной степенью
    thresholds = THRESHOLDS
    degree_shares = {}
    for t in thresholds:
        count = int(np.sum(degrees <= t))
        degree_shares[f"le_{t}"] = {
            "count": count,
            "share": round(count / n_nodes, 4)
        }
    stats["degree_shares"] = degree_shares

    # Распределение малых степеней (1...10)
    small_degrees = {
        str(k): int(np.sum(degrees == k))
        for k in range(1, 11)
    }
    stats["small_degrees"] = small_degrees

    # Вывод
    print(f"  Узлов:              {n_nodes:>12,}")
    print(f"  Рёбер:              {n_edges:>12,}")
    print(f"  Средняя степень:    {stats['mean_degree']:>12.2f}")
    print(f"  Медиана:            {stats['median_degree']:>12}")
    print(f"  Мода:               {stats['mode_degree']:>12}")
    print(f"  Макс. степень (хаб):{stats['max_degree']:>12,}")

    print("\n  Процентили:")
    for p, v in percentiles.items():
        print(f"    P{p}: {v:>5}")

    print("\n  Доля узлов со степенью ≤ T:")
    for t in thresholds:
        d = degree_shares[f"le_{t}"]
        print(f"    ≤ {t:>4}: {d['count']:>10,} ({d['share'] * 100:5.2f}%)")

    return stats


def topology_metrics(G: ig.Graph, sample_size: int = None) -> dict:
    """Топологические метрики: компоненты связности, кластеризация, средний путь.

    Средняя длина пути считается методом Монте-Карло (сэмплирование),
    чтобы не съесть оперативку на графах-миллионниках.
    """
    if sample_size is None:
        sample_size = config.MONTE_CARLO_SAMPLE_SIZE

    print("Топологические метрики")

    results = {}

    # 1. Компоненты связности
    print("\nПоиск компонент связности:")
    t0 = time.time()
    components = G.components()
    sizes = components.sizes()
    giant_size = max(sizes)
    giant_pct = giant_size / G.vcount() * 100

    results["components"] = {
        "n_components": len(sizes),
        "giant_component_size": giant_size,
        "giant_component_pct": round(giant_pct, 2),
    }
    print(f"  Всего компонент: {len(sizes)}")
    print(f"  Гигантская: {giant_size:,} ({giant_pct:.2f}%) - {time.time() - t0:.1f}с")

    # 2. Коэффициент кластеризации (глобальная транзитивность)
    print("\nКоэффициент кластеризации (транзитивность):")
    t0 = time.time()
    clustering = G.transitivity_undirected()
    results["clustering_coefficient"] = round(clustering, 6)
    print(f"  C = {clustering:.6f} - {time.time() - t0:.1f}с")

    # Для сравнения: ожидаемый C в случайном графе Эрдёша-Реньи
    # C_rand ~ <k> / N
    n = G.vcount()
    mean_k = 2 * G.ecount() / n
    c_rand = mean_k / n
    results["c_random"] = round(c_rand, 8)
    results["c_over_c_random"] = round(clustering / c_rand, 1)
    print(f"  C_rand ~ <k>/N = {c_rand:.2e}")
    print(f"  C/C_rand ~ {clustering / c_rand:.1f}x")

    # 3. Средняя длина пути (Monte Carlo на гигантской компоненте)
    print(f"\nСредняя длина пути (выборка {sample_size} узлов):")
    t0 = time.time()

    giant_id = int(np.argmax(sizes))
    G_giant = components.subgraph(giant_id)

    random.seed(config.RANDOM_SEED)
    sample_nodes = random.sample(range(G_giant.vcount()),
                                 min(sample_size, G_giant.vcount()))

    total_dist = 0
    count_paths = 0

    for i, node in enumerate(sample_nodes):
        paths = G_giant.distances(source=[node], target=None, weights=None)[0]
        total_dist += sum(paths)
        count_paths += len(paths)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{sample_size}")

    avg_path = total_dist / count_paths
    results["avg_path_length"] = round(avg_path, 3)

    # Теоретическое ожидание: <l> ~ ln(N) / ln(<k>)
    l_theor = np.log(n) / np.log(mean_k)
    results["avg_path_theoretical"] = round(l_theor, 3)

    print(f"  L = {avg_path:.3f} (теор. ~ {l_theor:.3f}) - {time.time() - t0:.1f}с")

    return results


def estimate_gamma_mle(data, k_min: int) -> dict:
    """Оценка показателя степенного закона γ через MLE.

    Формула:
        γ = 1 + n / Σ ln(k_i / (k_min - 0.5))

    Returns:
        dict с gamma, sigma (стандартная ошибка), n (размер выборки)
    """
    data = np.array(data)
    data = data[data >= k_min]
    n = len(data)

    if n == 0:
        return {"gamma": None, "sigma": None, "n": 0, "share": 0.0}

    gamma = 1 + n / np.sum(np.log(data / (k_min - 0.5)))
    sigma = np.sqrt((gamma - 1) ** 2 / n)

    return {
        "gamma": round(float(gamma), 3),
        "sigma": round(float(sigma), 3),
        "n": int(n),
    }


def power_law_analysis(G_undirected: ig.Graph, G_directed: ig.Graph = None) -> dict:
    """Полный анализ степенного закона: γ для in/out/und степеней."""
    print("Анализ степенного закона")

    results = {}
    n_total = G_undirected.vcount()

    # Неориентированная степень
    k_und = np.array(G_undirected.degree())
    k_min_und = config.POWER_LAW_K_MIN["und"]
    res_und = estimate_gamma_mle(k_und, k_min_und)
    res_und["share"] = round(res_und["n"] / n_total, 4)
    res_und["k_min"] = k_min_und
    results["undirected"] = res_und
    print(f"\nНеориентированная (k_min={k_min_und}):")
    print(f"  γ = {res_und['gamma']} ± {res_und['sigma']}")
    print(f"  n = {res_und['n']:,} ({res_und['share'] * 100:.1f}%)")

    # Ориентированные степени (если передан ориентированный граф)
    if G_directed is not None:
        # Входящая
        k_in = np.array(G_directed.indegree())
        k_min_in = config.POWER_LAW_K_MIN["in"]
        res_in = estimate_gamma_mle(k_in, k_min_in)
        res_in["share"] = round(res_in["n"] / n_total, 4)
        res_in["k_min"] = k_min_in
        results["in"] = res_in
        print(f"\nВходящая (k_min={k_min_in}):")
        print(f"  γ = {res_in['gamma']} ± {res_in['sigma']}")
        print(f"  n = {res_in['n']:,} ({res_in['share'] * 100:.1f}%)")

        # Исходящая
        k_out = np.array(G_directed.outdegree())
        k_min_out = config.POWER_LAW_K_MIN["out"]
        res_out = estimate_gamma_mle(k_out, k_min_out)
        res_out["share"] = round(res_out["n"] / n_total, 4)
        res_out["k_min"] = k_min_out
        results["out"] = res_out
        print(f"\nИсходящая (k_min={k_min_out}):")
        print(f"  γ = {res_out['gamma']} ± {res_out['sigma']}")
        print(f"  n = {res_out['n']:,} ({res_out['share'] * 100:.1f}%)")

    return results


def community_analysis(gt: dict) -> dict:
    """Анализ распределения размеров GT-сообществ по диапазонам."""
    print("Анализ GT-сообществ")

    sizes = [len(nodes) for nodes in gt.values()]
    n_cats = len(sizes)
    n_nodes = sum(sizes)

    results = {
        "n_categories": n_cats,
        "n_nodes_in_gt": n_nodes,
        "median_size": int(np.median(sizes)),
        "mode_size": int(Counter(sizes).most_common(1)[0][0]),
        "min_size": int(min(sizes)),
        "max_size": int(max(sizes)),
        "mean_size": round(float(np.mean(sizes)), 2),
    }

    print(f"Категорий:     {n_cats:>10,}")
    print(f"Узлов в GT:    {n_nodes:>10,}")
    print(f"Медиана:       {results['median_size']:>10}")
    print(f"Мода:          {results['mode_size']:>10}")
    print(f"Мин/Макс:      {results['min_size']:,} / {results['max_size']:,}")

    # Распределение по диапазонам
    print("\nРаспределение по диапазонам:")
    ranges_results = {}
    for name, (lo, hi) in config.SIZE_RANGES.items():
        if hi == float('inf'):
            cats_in_range = [s for s in sizes if s >= lo]
        else:
            cats_in_range = [s for s in sizes if lo <= s < hi]

        n_cats_range = len(cats_in_range)
        n_nodes_range = sum(cats_in_range)

        ranges_results[name] = {
            "n_categories": n_cats_range,
            "share_categories": round(n_cats_range / n_cats, 4),
            "n_nodes": n_nodes_range,
            "share_nodes": round(n_nodes_range / n_nodes, 4),
        }

        print(f"  {name:>6}: {n_cats_range:>6} кат. ({n_cats_range / n_cats * 100:5.1f}%) | "
              f"{n_nodes_range:>8} узлов ({n_nodes_range / n_nodes * 100:5.1f}%)")

    results["size_ranges"] = ranges_results
    return results

if __name__ == "__main__":
    print("Это вспомогательная функция анализа графа!")