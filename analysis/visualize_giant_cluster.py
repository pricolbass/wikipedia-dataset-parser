#!/usr/bin/env python3
"""
Визуализация внутренней структуры Гигантского кластера Leiden
+ Текстовая статистика "сошлось/не сошлось" по категориям.
"""
import igraph as ig
import pandas as pd
import random
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter, defaultdict


# Все нужные пути
GRAPH_PATH = "/Users/ligma/Desktop/Graphs/wikipedia-dataset-parser/ru_datasets/NMI compatible/Всё 1 млн/wiki_neutral_graph_20260401_141856.edgelist"
GT_PATH = "/Users/ligma/Desktop/Graphs/wikipedia-dataset-parser/ru_datasets/NMI compatible/Всё 1 млн/wiki_neutral_communities_20260401_141856.cmty"
OUTPUT_IMAGE = "giant_cluster_visualization.png"
OUTPUT_STATS = "giant_cluster_stats.txt"

SAMPLE_SIZE = 300 # Размер выборкм
RESOLUTION = 1.0


def load_graph():
    print("Загрузка графа:")
    df = pd.read_csv(GRAPH_PATH, sep='\t', header=None, names=['source', 'target'], dtype=str)
    df = df.dropna()
    df = df[df['source'] != df['target']]
    df = df.drop_duplicates()
    edges = list(zip(df['source'], df['target']))
    G = ig.Graph.TupleList(edges, directed=False, vertex_name_attr='name')
    print(f"  Граф: {G.vcount():,} вершин, {G.ecount():,} рёбер")
    return G


def load_gt_categories():
    print("Загрузка GT-категорий:")
    gt_map = {}
    with open(GT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                cat, article = parts[0], parts[1]
                if article not in gt_map:
                    gt_map[article] = cat
    print(f"  Загружено {len(gt_map):,} статей")
    return gt_map


def main():
    G = load_graph()
    gt_map = load_gt_categories()

    # 1. Запуск Leiden
    print(f"\nЗапуск Leiden (resolution={RESOLUTION}):")
    partition = G.community_leiden(objective_function='modularity', resolution=RESOLUTION, n_iterations=100)
    membership = partition.membership

    # 2. Находим гигантский кластер
    cluster_counts = Counter(membership)
    giant_cluster_id = cluster_counts.most_common(1)[0][0]
    giant_size = cluster_counts[giant_cluster_id]
    print(f"  Найден Гигантский кластер #{giant_cluster_id} (размер: {giant_size:,} узлов)")

    # 3. Берем случайную выборку из гигантского кластера
    nodes_in_giant = [v.index for v in G.vs if membership[v.index] == giant_cluster_id]
    if len(nodes_in_giant) < SAMPLE_SIZE:
        sample_nodes = nodes_in_giant
    else:
        sample_nodes = random.sample(nodes_in_giant, SAMPLE_SIZE)

    print(f"Создание визуализации выборки ({len(sample_nodes)} узлов):")

    # 4. Сошлось/не сошлось
    stats_report = []
    stats_report.append(f"Анализ выборки из гигантского кластера (ID={giant_cluster_id})")

    # Собираем статьи из выборки и их категории
    sample_articles = [G.vs[idx]['name'] for idx in sample_nodes]

    # Группируем по категориям
    cats_in_sample = defaultdict(list)
    for article in sample_articles:
        cat = gt_map.get(article, "Другое")
        cats_in_sample[cat].append(article)

    # Анализируем каждую категорию
    for cat, articles in sorted(cats_in_sample.items(), key=lambda x: len(x[1]), reverse=True):
        # Сколько статей этой категории попало в гигантский кластер
        # (все sample_articles уже оттуда, но проверим, не разбились ли они внутри выборки по мелким кластерам)

        cluster_ids_in_sample = [membership[G.vs['name'].index(a)] for a in articles]
        cluster_counter = Counter(cluster_ids_in_sample)

        # Основной кластер для этой категории внутри выборки
        main_sub_cluster, count = cluster_counter.most_common(1)[0]
        is_giant = (main_sub_cluster == giant_cluster_id)

        status = "Сошлось" if is_giant else "Разбилось"

        stats_report.append(f"Категория: '{cat}'")
        stats_report.append(f"  Всего в выборке: {len(articles)}")
        stats_report.append(f"  Статус: {status} (доминирующий под-кластер #{main_sub_cluster})")
        stats_report.append(f"  Примеры статей: {', '.join(articles[:5])}")
        stats_report.append(f"\n")

    # Сохраняем текстовый отчет
    with open(OUTPUT_STATS, 'w', encoding='utf-8') as f:
        f.write('\n'.join(stats_report))
    print(f"Текстовый отчет сохранен в {OUTPUT_STATS}")

    # Печатаем в консоль первые 5 категорий
    print("\nВывод первых 5 категорий")
    for line in stats_report[:25]:
        print(line)

    # 5. визуализация
    nx_g = nx.Graph()
    node_colors = []

    # Собираем уникальные категории для цветов
    unique_cats = set()
    for v_idx in sample_nodes:
        name = G.vs[v_idx]['name']
        cat = gt_map.get(name, "Другое")
        unique_cats.add(cat)

    # Генерируем цвета (чтобы было оч оч оч красиво)
    color_map = {cat: plt.cm.tab20(i % 20) for i, cat in enumerate(sorted(unique_cats))}

    for v_idx in sample_nodes:
        name = G.vs[v_idx]['name']
        cat = gt_map.get(name, "Другое")
        nx_g.add_node(name)
        node_colors.append(color_map[cat])

    # Добавляем рёбра
    sample_set = set(sample_nodes)
    for v_idx in sample_nodes:
        neighbors = G.neighbors(v_idx)
        for n_idx in neighbors:
            if n_idx in sample_set:
                nx_g.add_edge(G.vs[v_idx]['name'], G.vs[n_idx]['name'])

    # Рисуем
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(nx_g, k=0.5, iterations=100, seed=42)

    nx.draw_networkx_edges(nx_g, pos, alpha=0.1, edge_color='gray')
    nx.draw_networkx_nodes(nx_g, pos, node_size=50, node_color=node_colors, alpha=0.8)

    # Легенда 💪
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=cat,
                                  markerfacecolor=color_map[cat], markersize=10)
                       for cat in sorted(unique_cats)]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=8, title="GT Категории")

    plt.title(f"Структура Гигантского кластера Leiden (выборка {SAMPLE_SIZE} узлов)\n"
              f"Размер кластера: {giant_size:,} | Resolution: {RESOLUTION}", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=1000, bbox_inches='tight')
    plt.close()

    print(f"\nКартинка сохранена: {OUTPUT_IMAGE}")


if __name__ == '__main__':
    main()