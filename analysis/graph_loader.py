"""
Модуль загрузки графа Википедии и Ground Truth разметки.

Использование:
    from graph_loader import load_graph, load_ground_truth

    G = load_graph()                      # неориентированный
    G_dir = load_graph(directed=True)     # ориентированный
    gt = load_ground_truth()              # dict: {категория: [статьи]}
"""
import pandas as pd
import igraph as ig
from collections import defaultdict
from pathlib import Path

import config


def load_graph(edge_path: Path = None, directed: bool = False) -> ig.Graph:
    """Загружает граф Википедии из *.edgelist файла.

    Использует pandas для корректной обработки кириллицы и спецсимволов

    Args:
        edge_path: путь к .edgelist файлу. По умолчанию - из config.
        directed: если True, возвращает ориентированный граф.

    Returns:
        ig.Graph с атрибутом 'name' у вершин.
    """
    if edge_path is None:
        edge_path = config.EDGE_PATH

    print(f"Загрузка графа: {edge_path}")
    print(f"Режим: {'ориентированный' if directed else 'неориентированный'}")

    df = pd.read_csv(
        edge_path,
        sep='\t',
        header=None,
        names=['source', 'target'],
        dtype=str,
        low_memory=False
    )
    n_raw = len(df)

    # Очистка: NaN, самопетли, дубликаты
    df = df.dropna()
    df = df[df['source'] != df['target']]
    df = df.drop_duplicates()

    print(f"  Сырых рёбер: {n_raw:,}")
    print(f"  После очистки: {len(df):,}")

    edges = list(zip(df['source'], df['target']))
    G = ig.Graph.TupleList(
        edges,
        directed=directed,
        vertex_name_attr='name'
    )

    print(f"  Граф: {G.vcount():,} вершин, {G.ecount():,} рёбер")
    return G


def load_ground_truth(gt_path: Path = None) -> dict:
    """Загружает GT-разметку из *.cmty файла.

    Формат файла: категория<TAB>узел

    Применяется правило "первая тема":
    если статья встречается в нескольких категориях,
    сохраняется только первая встреча.

    Args:
        gt_path: путь к .cmty файлу. По умолчанию — из config.

    Returns:
        dict: {название_категории: [список_статей]}
    """
    if gt_path is None:
        gt_path = config.GT_PATH

    print(f"Загрузка GT: {gt_path}")

    communities = defaultdict(list)
    with open(gt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                cat, article = parts[0], parts[1]
                communities[cat].append(article)

    n_cats = len(communities)
    n_nodes = sum(len(nodes) for nodes in communities.values())
    print(f"  Загружено: {n_cats:,} категорий, {n_nodes:,} статей")
    return dict(communities)


def get_gt_article_to_category(gt: dict) -> dict:
    """Преобразует {категория: [статьи]} -> {статья: категория}.

    Применяет правило "первая тема": каждая статья относится
    к первой категории, в которой она встретилась.
    """
    article_to_cat = {}
    for cat, articles in gt.items():
        for article in articles:
            if article not in article_to_cat:
                article_to_cat[article] = cat
    return article_to_cat


"""Хелперы"""

def get_degrees(G: ig.Graph, mode: str = "all") -> list:
    """Возвращает список степеней всех вершин.

    Args:
        G: граф
        mode: "in", "out" или "all"
    """
    return G.degree(mode=mode)


if __name__ == "__main__":
    print("Это вспомогательная функция загрузки графа!")