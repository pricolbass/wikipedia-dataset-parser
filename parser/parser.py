import bz2
import re
import sys
import igraph as ig
import pandas as pd
import json
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import warnings
import os


"""
Флаг 'NMI = True' делает возможным использование одноимённой метрики. Присваивая статье, у которой несколько категорий самую первую
'NMI = False' делает возможным запись одной статье нескольких категорий

Флаг 'ALL = True' отключает поиск по целевым статьям, которые задаются ниже (если цель отфильтровать только статьи про биологию - отлично!)
'ALL = False' записывает ВСЕ статьи, исключая только статьи/категории из чёрного списка
"""

NMI = True
ALL = False

# Лимит на максимальное кол-во статей (sys.maxsize выдаёт огромное число, что значит, что ВСЕ статьи будут обработаны)
max_articles = sys.maxsize
print(f"Ограничение по кол-во статей: {max_articles}\n")

blacklist_file = "list/black_list.txt"

class WikipediaDumpParser:
    def __init__(self, max_articles: int = 50000, blacklist_file: str = "blacklist.txt"):
        self.max_articles = max_articles

        # Загружаем чёрные списки СТРАНИЦ
        self.blacklist_pages = self._load_blacklist(blacklist_file)

        # Чёрный список СТАТЕЙ
        self.blacklist_categories = [
            "Оружие", "Война", "Преступления", "Насилие", "Смерть",
            "Терроризм", "Экстремизм", "Порнография", "Наркотики",
            "Самоубийства", "Аборты", "Эвтаназия", "Секты",
            "Политические репрессии", "Геноцид", "Холокост",
            "Военные преступления", "Пытки", "Изнасилования",
            "Пропаганда", "Цензура", "Манипуляция", "Война",
        ]

        # Фильтр по целевым категориям
        self.target_categories = [
            "Биология", "Ботаника", "Зоология", "Генетика", "Экология",
            "Таксономия", "Виды", "Роды (биология)", "Семейства (биология)",
            "Отряды (биология)", "Классы (биология)", "Типы (биология)",
            "Царства (биология)", "Млекопитающие", "Птицы", "Рыбы",
            "Насекомые", "Паукообразные", "Моллюски", "Черви", "Грибы",
            "Бактерии", "Вирусы", "Растения", "Животные", "Протисты",
            "Археи", "Анатомия", "Физиология", "Биохимия",
            "Молекулярная биология", "Клеточная биология",
            "Эволюционная биология", "Микробиология", "Иммунология",
            "Нейробиология", "Биофизика", "Биологические термины",
            "Математика", "Физика", "Химия", "Астрономия", "Геология",
            "География", "Лингвистика", "Философия", "Искусство",
            "Литература", "Музыка", "Архитектура", "Технологии",
        ]
        if ALL:
            self.target_categories = None

            # Ключевые слова для нейтральных наук
        self.science_keywords = [
            'биолог', 'ботан', 'зоолог', 'генетик', 'эколог',
            'таксон', 'вид', 'род', 'семейств', 'отряд', 'класс',
            'тип', 'царств', 'млекопита', 'птиц', 'рыб', 'насеком',
            'паук', 'моллюск', 'черв', 'гриб', 'бактери', 'вирус',
            'растен', 'животн', 'протист', 'архей', 'анатом',
            'физиол', 'биохим', 'молекулярн', 'клеточн', 'эволюц',
            'микробиол', 'иммунол', 'нейробиол', 'биофизик',
            'флора', 'фауна', 'организм', 'клетка', 'ген', 'днк', 'рнк',
            'белок', 'фермент', 'метаболизм', 'фотосинтез',
            # Другие науки
            'математ', 'физик', 'химич', 'астроном', 'геолог',
            'географ', 'лингвист', 'философ', 'искусств', 'литератур',
            'музык', 'архитект', 'технолог', 'программ', 'алгоритм',
        ]
        if ALL:
            self.science_keywords = None

        self.articles = {}
        self.categories = defaultdict(list)
        self.id_to_category = {}
        self.edges = set()
        self.stats = {
            'pages': 0, 'articles': 0, 'edges': 0,
            'filtered_black_page': 0, 'filtered_black_cat': 0,
            'filtered_cat': 0, 'filtered_links': 0,
        }
        self.id_to_all_categories = defaultdict(list)

    def _load_blacklist(self, filepath: str) -> set:
        """Загружает чёрный список страниц из файла"""
        blacklist = set()
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        blacklist.add(line)
            print(f"Загружено {len(blacklist)} запрещённых страниц")
        else:
            warnings.warn(f"Файл {filepath} не найден, чёрный список страниц пуст\n\tЧёрный список категорий (blacklist_categories) всё ещё на месте!")
        return blacklist

    def extract_title(self, line: str) -> str:
        match = re.search(r'<title>(.*?)</title>', line)
        return match.group(1) if match else None

    def extract_links(self, text: str) -> list:
        links = re.findall(r'\[\[([^\]|]+)', text)
        filtered = []
        for link in links:
            link = link.strip()
            if ':' not in link:
                filtered.append(link)
        return filtered

    def extract_category(self, text: str) -> list:
        cats = re.findall(r'\[\[([Кк]атегория|[Cc]ategory):([^\]]+)\]\]', text)
        return [c[1].strip() for c in cats]

    def _is_blacklisted_page(self, title: str) -> bool:
        """Проверка названия статьи по чёрному списку"""
        # Точное совпадение
        if title in self.blacklist_pages:
            return True

        # Проверка по подстроке (для составных названий)
        title_lower = title.lower()
        for black in self.blacklist_pages:
            if black.lower() in title_lower:
                return True

        return False

    def _is_blacklisted_category(self, article_categories: list) -> bool:
        """Проверка категорий по чёрному списку"""
        for cat in article_categories:
            cat_lower = cat.lower()
            for black in self.blacklist_categories:
                if black.lower() in cat_lower:
                    return True
        return False

    def _is_target_topic(self, article_categories: list) -> bool:
        """Проверка на принадлежность к целевым темам (если таковые заданы)"""
        if not article_categories:
            return False

        for cat in article_categories:
            cat_lower = cat.lower()

            # 1. Точное совпадение с целевыми категориями
            if self.target_categories is not None:
                if cat in self.target_categories:
                    return True
            else:
                return True

            # 2. Проверка по ключевым словам
            if self.science_keywords is not None:
                for keyword in self.science_keywords:
                    if keyword in cat_lower:
                        return True
            else:
                return True

        return False

    def parse_xml_dump(self, filepath: str):
        print(f"Парсинг: {filepath}")
        print(f"Чёрный список: {len(self.blacklist_pages)} страниц + {len(self.blacklist_categories)} категории\n")

        current_title = None
        current_text = []
        in_page = False
        in_text = False

        with bz2.open(filepath, 'rt', encoding='utf-8') as f:
            for line in tqdm(f, desc="Чтение"):
                if '<page>' in line:
                    in_page = True
                    current_text = []
                    continue

                if in_page and '<title>' in line:
                    title = self.extract_title(line)
                    if title and ':' not in title:
                        current_title = title
                    else:
                        current_title = None
                    continue

                if in_page and '<text' in line:
                    in_text = True
                    continue

                if in_page and '</text>' in line:
                    in_text = False
                    continue

                if in_page and '</page>' in line:
                    if current_title and current_text:
                        self._process_article(current_title, ''.join(current_text))

                    in_page = False
                    current_title = None
                    current_text = []

                    if self.stats['articles'] >= self.max_articles:
                        print(f"\nЛимит: {self.max_articles} статей\n")
                        break
                    continue

                if in_text and current_title:
                    current_text.append(line)

        print(f"Парсинг завершён: {self.stats['articles']} статей")
        print(f"Отфильтровано (чёрный список страниц): {self.stats['filtered_black_page']}")
        print(f"Отфильтровано (чёрный список категорий): {self.stats['filtered_black_cat']}")
        print(f"Отфильтровано (не целевая тема): {self.stats['filtered_cat']}\n")

    def _process_article(self, title: str, text: str):
        self.stats['pages'] += 1

        # Фильтруем очень длинные названия
        if len(text) < 100:
            return

        # Проверка названия
        if self._is_blacklisted_page(title):
            self.stats['filtered_black_page'] += 1
            return # Отбрасываем статью

        cats = self.extract_category(text)

        # Проверка категорий
        if self._is_blacklisted_category(cats):
            self.stats['filtered_black_cat'] += 1
            return

        # Фильтрация ссылок
        links = self.extract_links(text)
        for link in links:
            if self._is_blacklisted_page(link):
                self.stats['filtered_links'] += 1
                return

        # Проверка на целевую тему
        if not self._is_target_topic(cats):
            self.stats['filtered_cat'] += 1
            return

        # Статья прошла все фильтры
        self.stats['articles'] += 1
        article_id = self.stats['articles'] - 1
        self.articles[title] = article_id
        self.id_to_category[article_id] = cats[0] if cats else 'uncategorized'

        # Проверка на флаг NMI
        if not NMI:
            self.id_to_all_categories[article_id] = cats

        # Сохраняем только безопасные ссылки
        safe_links = [l for l in links if not self._is_blacklisted_page(l)]
        for link in safe_links:
            if link != title:
                self.edges.add((title, link))
                self.stats['edges'] += 1

        for cat in cats:
            self.categories[cat].append(article_id)

    def build_graph(self) -> ig.Graph:
        """Строит граф"""
        print(f"\nПостроение графа...")

        article_titles = set(self.articles.keys())
        valid_edges = []

        print(f"Валидация рёбер ({len(self.edges)})...")
        for src, dst in tqdm(self.edges, desc="Валидация"):
            if src in article_titles and dst in article_titles:
                src_id = self.articles[src]
                dst_id = self.articles[dst]
                valid_edges.append((src_id, dst_id))

        print(f"Создание графа...")
        G = ig.Graph(n=len(self.articles), edges=valid_edges, directed=False)

        print(f"Атрибуты узлов...")
        id_to_title = {v: k for k, v in self.articles.items()}

        G.vs['name'] = [id_to_title.get(v.index, f'unknown_{v.index}') for v in G.vs]
        G.vs['primary_category'] = [self.id_to_category.get(v.index, 'uncategorized') for v in G.vs]

        print(f"Удаление изолированных...")
        G = G.subgraph([v.index for v in G.vs if v.degree() > 0])

        print(f"\nГраф: {G.vcount()} узлов, {G.ecount()} рёбер")
        return G

    def save_results(self, G: ig.Graph, communities: dict, output_dir: str = "data_out"):
        """Сохранение"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_dir = f"{output_dir}/wiki_parse_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        # Граф
        edges_with_names = []
        for src_id, dst_id in G.get_edgelist():
            src_name = G.vs[src_id]['name']
            dst_name = G.vs[dst_id]['name']
            edges_with_names.append((src_name, dst_name))

        edges_df = pd.DataFrame(edges_with_names, columns=['source', 'target'])
        graph_file = f"{output_dir}/wiki_neutral_graph_{timestamp}.edgelist"
        edges_df.to_csv(graph_file, sep='\t', index=False, header=False, encoding='utf-8')
        print(f"Граф: {graph_file}")

        # Проверка первых 3-х строк файла на сепаратор
        with open(graph_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < 3:
                    tabs = line.count('\t')
                    print(f"Строка {i}: {tabs} табуляций {'GOOD' if tabs == 1 else warnings.warn('Табуляция сломалась!')}")

        # Ground Truth
        id_to_title = {v: k for k, v in self.articles.items()}
        if NMI:
            cmty_file = f"{output_dir}/wiki_neutral_communities_{timestamp}.cmty"
            with open(cmty_file, 'w', encoding='utf-8') as f:
                for cat, articles in communities.items():
                    safe_cat = cat.replace(' ', '_').replace('/', '_')
                    for article in articles:
                        f.write(f"{safe_cat}\t{article}\n")
            print(f"GT: {cmty_file}")
        else:
            cmty_file = f"{output_dir}/wiki_neutral_communities_{timestamp}.cmty"
            with open(cmty_file, 'w', encoding='utf-8') as f:
                for article_id, cats in self.id_to_all_categories.items():
                    article_name = id_to_title[article_id]  # ← Теперь работает!
                    for cat in cats:
                        safe_cat = cat.replace(' ', '_').replace('/', '_')
                        f.write(f"{safe_cat}\t{article_name}\n")

        # Статистика
        stats = {
            'nodes': G.vcount(),
            'edges': G.ecount(),
            'communities': len(communities),
            'blacklist_pages': len(self.blacklist_pages),
            'blacklist_categories': len(self.blacklist_categories),
            'filtered_black_page': self.stats['filtered_black_page'],
            'filtered_black_cat': self.stats['filtered_black_cat'],
            'filtered_other': self.stats['filtered_cat'],
            'timestamp': timestamp
        }
        stats_file = f"{output_dir}/wiki_neutral_stats_{timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"Stats: {stats_file}")

        return timestamp

    def extract_ground_truth(self, G: ig.Graph) -> dict:
        """GT из категорий"""
        communities = {}

        for v in G.vs:
            cat = v['primary_category']
            communities.setdefault(cat, []).append(v['name'])

        # Фильтр мелких сообществ (минимум 3 статьи)
        communities = {k: v for k, v in communities.items() if len(v) >= 3}
        print(f"Сообществ: {len(communities)}")
        return communities


if ALL:
    warnings.warn(f"\nФлаг ALL стоит в положении True!\nФильтра по целевым статьям НЕ БУДЕТ!\n")
else:
    print(f"Фильтр по целевым категориям включен!\nЕсли не хотите такой фильтр, сделайте флаг ALL = False\n")

if NMI:
    print(f"Защита для NMI включена!\n")
else:
    warnings.warn(f"\nФлаг NMI = False!\nОдной статье может принадлежать много категорий!")

parser = WikipediaDumpParser(
    max_articles,
    blacklist_file
)

parser.parse_xml_dump("wiki_data/ruwiki-latest-pages-articles.xml.bz2")

G = parser.build_graph()

communities = parser.extract_ground_truth(G)

parser.save_results(G, communities)

print(f"{G.vcount()} узлов, {G.ecount()} рёбер, {len(communities)} сообществ")