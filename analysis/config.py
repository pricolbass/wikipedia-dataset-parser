"""Конфигурация проекта: пути, константы, параметры анализа."""
from pathlib import Path

# Пути к данным
DATA_DIR = Path("/Users/ligma/Desktop/Graphs/wikipedia-dataset-parser/ru_datasets/NMI compatible/Всё 1 млн")
EDGE_PATH = DATA_DIR / "wiki_neutral_graph_20260401_141856.edgelist"
GT_PATH = DATA_DIR / "wiki_neutral_communities_20260401_141856.cmty"

# Пути для вывода
OUTPUT_DIR = Path("./output")
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

# Создаём директории при импорте
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

"""Параметры анализа"""

# Пороги k_min для оценки γ (gamma)
POWER_LAW_K_MIN = {
    "in": 5,
    "out": 3,
    "und": 4,
}

# Размерные диапазоны для многоуровневой оценки
SIZE_RANGES = {
    "3-10":  (3, 10),
    "10-20": (10, 20),
    "20-50": (20, 50),
    "50+":   (50, float('inf')),
}

# Процентили
PERCENTILES = [25, 50, 75, 90, 95, 99]

# Разная степень узлов
THRESHOLDS = [1, 2, 3, 5, 10, 20, 50, 100, 1000]

# Параметры Monte-Carlo оценки средней длины пути
MONTE_CARLO_SAMPLE_SIZE = 2000
RANDOM_SEED = 892

# Визуализация
FIGURE_DPI = 300

"""Параметры Leiden"""
# Значения параметра гранулярности для перебора
LEIDEN_RESOLUTIONS = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]

# Ограничение итераций Leiden
# 10 - быстрый тест, 30 - оптимальный баланс, 100 - финальный запуск
LEIDEN_N_ITERATIONS = 100

# Директория для результатов Leiden
LEIDEN_RESULTS_DIR = OUTPUT_DIR / "leiden"
LEIDEN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print("Это конфиг, его запускать не надо")