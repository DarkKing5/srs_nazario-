import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter # Для подсчета частоты жанров
import numpy as np

# Устанавливаем стиль для графиков (опционально)
sns.set_theme(style="whitegrid")

def plot_genre_distribution(recommended_df, genre_list_col='genre_list', top_n=15, title="Распределение жанров в рекомендациях"):
    """
    Строит гистограмму распределения жанров среди рекомендованных книг.

    Args:
        recommended_df (pd.DataFrame): DataFrame, содержащий ТОЛЬКО рекомендованные книги.
                                       Должен содержать колонку со списками жанров.
        genre_list_col (str): Название колонки, содержащей списки жанров (например, 'genre_list').
        top_n (int): Сколько самых популярных жанров отображать.
        title (str): Заголовок графика.
    """
    if genre_list_col not in recommended_df.columns:
        print(f"Ошибка: Колонка '{genre_list_col}' не найдена в DataFrame рекомендованных книг.")
        return

    print(f"\nПостроение графика: {title}")
    # Собираем все жанры из списков в один большой список
    all_genres = [genre for sublist in recommended_df[genre_list_col] for genre in sublist if sublist] # Проверяем, что sublist не пустой

    if not all_genres:
        print("Нет жанров для отображения.")
        return

    # Считаем частоту каждого жанра
    genre_counts = Counter(all_genres)

    # Отбираем top_n самых частых жанров
    common_genres = genre_counts.most_common(top_n)

    if not common_genres:
        print("Не удалось определить самые частые жанры.")
        return

    # Разделяем жанры и их количество для графика
    genres, counts = zip(*common_genres)

    # Строим график
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(counts), y=list(genres), palette="viridis")
    plt.title(title)
    plt.xlabel("Количество книг")
    plt.ylabel("Жанр")
    plt.tight_layout() # Чтобы подписи не перекрывались
    plt.show()

def plot_similarity_distribution(sim_scores, book_title, self_index):
    """
    Строит гистограмму распределения оценок сходства для данной книги.

    Args:
        sim_scores (list): Список кортежей (index, similarity_score) для одной книги.
        book_title (str): Название книги, для которой строится график.
        self_index (int): Индекс самой книги (чтобы исключить ее оценку 1.0).
    """
    print(f"\nПостроение графика: Распределение сходства для '{book_title}'")
    # Извлекаем только оценки сходства, исключая оценку для самой книги (которая равна 1.0)
    scores = [score for idx, score in sim_scores if idx != self_index]

    if not scores:
        print("Нет оценок сходства для отображения.")
        return

    plt.figure(figsize=(10, 5))
    sns.histplot(scores, bins=50, kde=True) # kde=True добавляет сглаженную кривую
    plt.title(f"Распределение косинусного сходства с книгой\n'{book_title}'")
    plt.xlabel("Косинусное сходство")
    plt.ylabel("Количество книг")
    plt.show()

def plot_rating_distribution(recommended_df, rating_col='Avg_Rating', title="Распределение рейтингов в рекомендациях"):
    """
    Строит гистограмму распределения средних рейтингов среди рекомендованных книг.

    Args:
        recommended_df (pd.DataFrame): DataFrame, содержащий ТОЛЬКО рекомендованные книги.
                                       Должен содержать колонку с рейтингами.
        rating_col (str): Название колонки с рейтингами (например, 'Avg_Rating').
        title (str): Заголовок графика.
    """
    if rating_col not in recommended_df.columns:
        print(f"Ошибка: Колонка '{rating_col}' не найдена в DataFrame рекомендованных книг.")
        return

    print(f"\nПостроение графика: {title}")
    plt.figure(figsize=(10, 5))
    sns.histplot(recommended_df[rating_col], bins=10, kde=False)
    plt.title(title)
    plt.xlabel("Средний рейтинг")
    plt.ylabel("Количество книг")
    plt.xlim(0, 5) # Ограничим ось X стандартным диапазоном рейтинга
    plt.show()

def plot_evaluation_metrics(metrics_dict, title="Метрики качества рекомендаций"):
    """
    Строит гистограмму для отображения метрик оценки (Precision, Recall, F1).

    Args:
        metrics_dict (dict): Словарь с метриками, например:
                             {'Precision@K': 0.7, 'Recall@K': 0.6, 'F1@K': 0.65}
        title (str): Заголовок графика.
    """
    if not metrics_dict:
        print("Нет метрик для отображения.")
        return

    print(f"\nПостроение графика: {title}")
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values, color=['skyblue', 'lightgreen', 'salmon'])
    # Добавляем значения над столбцами
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom', ha='center') # va: vertical alignment

    plt.title(title)
    plt.ylabel("Значение метрики")
    plt.ylim(0, 1.1) # Ось Y от 0 до 1.1 (чуть выше 1 для подписей)
    plt.show()

# Пример использования (как вызывать эти функции)
# if __name__ == '__main__':
#     # --- Нужны тестовые данные ---
#     # 1. DataFrame с рекомендованными книгами (recommended_books_df)
#     #    с колонками 'genre_list' и 'Avg_Rating'
#     # 2. Список оценок сходства для одной книги (sim_scores_example)
#     # 3. Словарь с посчитанными метриками (metrics_example)
#
#     # --- Начало блока ЗАГЛУШЕК (для примера) ---
#     dummy_rec_data = {
#         'Book': ['Rec Book 1', 'Rec Book 2', 'Rec Book 3'],
#         'genre_list': [['Fantasy', 'Magic'], ['Fantasy', 'Adventure'], ['Classics']],
#         'Avg_Rating': [4.8, 4.5, 3.9]
#     }
#     recommended_books_df_example = pd.DataFrame(dummy_rec_data)
#
#     sim_scores_example = [(0, 1.0), (1, 0.9), (2, 0.5), (3, 0.8), (4, 0.2)]
#     book_title_example = "Тестовая Книга"
#     self_index_example = 0
#
#     metrics_example = {'Precision@3': 0.667, 'Recall@3': 0.667, 'F1@3': 0.667}
#     # --- Конец блока ЗАГЛУШЕК ---
#
#     # --- Вызов функций ---
#     plot_genre_distribution(recommended_books_df_example, top_n=5)
#     plot_similarity_distribution(sim_scores_example, book_title_example, self_index_example)
#     plot_rating_distribution(recommended_books_df_example)
#     plot_evaluation_metrics(metrics_example)
#