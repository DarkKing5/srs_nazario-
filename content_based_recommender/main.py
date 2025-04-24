import pandas as pd
import time
import nltk
import os
import sys  # Для замера времени выполнения (опционально)
# --- NLTK Data Path Configuration ---
# Определяем абсолютный путь к папке nltk_data внутри нашего проекта
# __file__ это путь к main.py
script_dir = os.path.dirname(os.path.abspath(__file__)) # Папка, где лежит main.py
# Поднимаемся на уровень выше (в srs_nazario) и добавляем nltk_data
nltk_data_dir = os.path.join(os.path.dirname(script_dir), 'nltk_data')

# Проверяем, существует ли папка и добавляем путь в NLTK, если да
if os.path.exists(nltk_data_dir):
    print(f"Найден путь NLTK Data: {nltk_data_dir}")
    # Добавляем наш путь в НАЧАЛО списка путей NLTK
    # чтобы он искал сначала здесь
    nltk.data.path.insert(0, nltk_data_dir)
else:
    print(f"Предупреждение: Папка nltk_data НЕ НАЙДЕНА по пути: {nltk_data_dir}")
    print("Пожалуйста, создайте папку nltk_data в корне проекта и скачайте туда 'punkt'")

# Распечатаем пути, где NLTK будет искать данные (для отладки)
# print(f"Текущие пути поиска NLTK: {nltk.data.path}")
# --- End NLTK Path Configuration ---

# !!! Далее идут ваши остальные импорты !!!
# import pandas as pd
# from .data_loader.loader import DataLoader ... и т.д.
# Импортируем классы и модули из нашей библиотеки
# Обратите внимание на точки в начале - это относительный импорт внутри пакета
# Импортируем классы и модули из нашей библиотеки
# Используем ТОЛЬКО относительный импорт внутри пакета
from .data_loader.data_loading import BookDataLoader
from .preprocessing.feature_engineer import FeatureEngineer
from .recommender.content_recommender import ContentRecommender
from .evaluation import metrics
from .visualization import plot_results

# --- Параметры ---
# Путь к файлу данных относительно main.py
# (main.py в content_based_recommender, data на уровень выше)
DATA_PATH = 'data/goodreads_data.csv'
NUM_RECOMMENDATIONS = 10 # Сколько рекомендаций выводить
RATING_WEIGHT = 0.1      # Вес рейтинга (0=только схожесть, 1=только рейтинг)
EVALUATION_K = 10        # Значение K для метрик Precision@K, Recall@K
# Пример книги для тестирования (возьмем из вашего примера данных)
EXAMPLE_BOOK_TITLE = "To Kill a Mockingbird"

# --- Флаги для опциональных шагов ---
RUN_EVALUATION = True    # Рассчитать и показать метрики?
RUN_VISUALIZATION = True # Показать графики?

# ========================
# Основной код пайплайна
# ========================
if __name__ == "__main__":

    start_time = time.time()

    # --- 1. Загрузка данных ---
    print("-" * 30)
    print("ШАГ 1: Загрузка данных...")
    try:
        loader = BookDataLoader(DATA_PATH)
        raw_df = loader.load_data()
    except FileNotFoundError as e:
        print(e)
        raw_df = None # Устанавливаем в None, чтобы последующие шаги не выполнялись

    if raw_df is None:
        print("\nНе удалось загрузить данные. Выполнение прервано.")
        exit() # Завершаем скрипт, если данные не загружены
    print(f"Загружено {raw_df.shape[0]} записей.")
    print("-" * 30)


    # --- 2. Инжиниринг признаков ---
    print("ШАГ 2: Обработка и векторизация признаков...")
    fe = FeatureEngineer()
    # Передаем копию, чтобы не изменять raw_df
    processed_df = fe.preprocess_dataframe(raw_df.copy())
    fe.fit_transform_features(processed_df)
    feature_matrix = fe.get_combined_features(include_genres=True, include_description=True)

    if feature_matrix is None:
        print("\nНе удалось создать матрицу признаков. Выполнение прервано.")
        exit()
    print("-" * 30)


    # --- 3. Создание Рекомендателя ---
    print("ШАГ 3: Инициализация рекомендательной системы...")
    try:
        # Передаем processed_df т.к. он содержит доп. колонки вроде genre_list
        # и нормализованные рейтинги fe.normalized_avg_ratings
        recommender = ContentRecommender(processed_df, feature_matrix, fe.normalized_avg_ratings)
    except ValueError as e:
        print(f"Ошибка при создании рекомендателя: {e}")
        exit()
    print("Рекомендатель готов.")
    print("-" * 30)


    # --- 4. Получение Рекомендаций ---
    print("ШАГ 4: Генерация рекомендаций...")
    # Запрашиваем название книги у пользователя
    input_title = input(f"Введите название книги (или нажмите Enter для примера '{EXAMPLE_BOOK_TITLE}'): ")
    if not input_title:
        input_title = EXAMPLE_BOOK_TITLE

    recommendations = recommender.recommend(input_title, NUM_RECOMMENDATIONS, RATING_WEIGHT)

    if recommendations:
        print("\n--- Рекомендованные книги: ---")
        for i, book in enumerate(recommendations):
            print(f"{i+1}. {book}")
        print("-" * 30)
    else:
        print(f"\nНе удалось найти рекомендации для '{input_title}'.")
        # Если книга не найдена, нет смысла делать оценку/визуализацию для нее
        RUN_EVALUATION = False
        RUN_VISUALIZATION = False
        print("-" * 30)


    # --- 5. Оценка (Опционально) ---
    if RUN_EVALUATION and recommendations: # Оцениваем, только если книга найдена и есть рекомендации
        print("ШАГ 5: Оценка качества (прокси)...")
        book_index = recommender.title_to_index.get(input_title)

        if book_index is not None:
            # Получаем "релевантные" - top K по чистому сходству
            sim_scores_enum = list(enumerate(recommender.cosine_sim[book_index]))
            relevant_indices = metrics.get_top_k_similar(sim_scores_enum, EVALUATION_K, book_index)

            # Получаем индексы рекомендованных книг
            # Нужно найти индексы для названий из списка recommendations
            recommended_indices = set(recommender.title_to_index[title] for title in recommendations if title in recommender.title_to_index)

            # Считаем метрики
            p_at_k = metrics.precision_at_k(recommended_indices, relevant_indices, EVALUATION_K)
            r_at_k = metrics.recall_at_k(recommended_indices, relevant_indices) # relevant_indices здесь имеет размер K
            f1 = metrics.f1_at_k(p_at_k, r_at_k)

            print(f"\nМетрики для '{input_title}' (K={EVALUATION_K}):")
            print(f"  Precision@{EVALUATION_K}: {p_at_k:.4f}")
            print(f"  Recall@{EVALUATION_K}:    {r_at_k:.4f}")
            print(f"  F1-score@{EVALUATION_K}:  {f1:.4f}")

            # Сохраняем для графика
            metrics_data = {f'Precision@{EVALUATION_K}': p_at_k,
                            f'Recall@{EVALUATION_K}': r_at_k,
                            f'F1@{EVALUATION_K}': f1}
        else:
            print("Не удалось найти индекс для оценки.")
            metrics_data = None
        print("-" * 30)


    # --- 6. Визуализация (Опционально) ---
    if RUN_VISUALIZATION:
        print("ШАГ 6: Визуализация результатов...")
        book_index = recommender.title_to_index.get(input_title) # Получаем индекс еще раз (или используем предыдущий)

        # Графики, связанные с рекомендациями (строим, если они есть)
        if recommendations:
            # Получаем DataFrame только с рекомендованными книгами
            recommended_books_df = processed_df[processed_df['Book'].isin(recommendations)]
            # Строим распределение жанров
            plot_results.plot_genre_distribution(recommended_books_df, title=f"Распределение жанров для '{input_title}'")
            # Строим распределение рейтингов
            plot_results.plot_rating_distribution(recommended_books_df, title=f"Распределение рейтингов для '{input_title}'")

        # Графики, связанные с самой книгой (строим, если книга найдена)
        if book_index is not None:
            # Строим распределение сходства
            sim_scores_enum = list(enumerate(recommender.cosine_sim[book_index]))
            plot_results.plot_similarity_distribution(sim_scores_enum, input_title, book_index)

        # График с метриками (строим, если считали)
        if RUN_EVALUATION and metrics_data:
            plot_results.plot_evaluation_metrics(metrics_data, title=f"Метрики для '{input_title}' (K={EVALUATION_K})")

        print("Визуализация завершена (окна с графиками могут быть позади).")
        print("-" * 30)


    # --- Завершение ---
    end_time = time.time()
    print(f"Выполнение скрипта завершено за {end_time - start_time:.2f} секунд.")