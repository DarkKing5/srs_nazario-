import pandas as pd
import numpy as np
import ast  # Для безопасного парсинга строки со списком жанров
import re   # Для очистки Num_Ratings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from scipy.sparse import hstack # Для объединения разреженных матриц (TF-IDF и жанры)

# Относительный импорт функции очистки текста из соседнего файла
try:
    from .text_cleaner import clean_text
except ImportError:
    # Запасной вариант, если запускаем файл напрямую для тестов
    from text_cleaner import clean_text

class FeatureEngineer:
    """
    Класс для инжиниринга признаков: очистка, обработка,
    векторизация и нормализация данных о книгах.
    """
    def __init__(self):
        """Инициализирует векторизаторы и скейлеры."""
        # max_features ограничивает количество самых частых слов для TF-IDF
        # ngram_range=(1, 2) учитывает одиночные слова и пары слов
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.mlb = MultiLabelBinarizer()
        self.rating_scaler = MinMaxScaler()
        self.num_rating_scaler = MinMaxScaler() # Отдельный скейлер для количества оценок

        # Атрибуты для хранения результатов
        self.processed_df = None
        self.description_matrix = None
        self.genre_matrix = None
        self.normalized_avg_ratings = None
        self.normalized_num_ratings = None
        self.feature_names = None # Для хранения имен признаков TF-IDF

    def _parse_genres(self, genres_str):
        """Безопасно парсит строку вида "['Genre1', 'Genre2']" в список."""
        if pd.isna(genres_str):
            return [] # Возвращаем пустой список для NaN
        try:
            # Используем literal_eval для безопасного выполнения строки как кода Python
            genres_list = ast.literal_eval(genres_str)
            # Убедимся, что результат - это список строк
            if isinstance(genres_list, list) and all(isinstance(g, str) for g in genres_list):
                return genres_list
            else:
                return [] # Некорректный формат внутри строки
        except (ValueError, SyntaxError, TypeError):
            # В случае ошибки парсинга возвращаем пустой список
            return []

    def _clean_num_ratings(self, num_ratings_str):
        """Очищает строку с числом рейтингов (удаляет запятые) и конвертирует в число."""
        if pd.isna(num_ratings_str):
            return 0 # Возвращаем 0 для NaN
        try:
            # Удаляем запятые и преобразуем в целое число
            return int(str(num_ratings_str).replace(',', ''))
        except ValueError:
             # Если не получается преобразовать, возвращаем 0
            return 0

    def preprocess_dataframe(self, df, text_col='Description', genre_col='Genres',
                             rating_col='Avg_Rating', num_rating_col='Num_Ratings'):
        """
        Выполняет предварительную обработку колонок DataFrame.

        Args:
            df (pd.DataFrame): Исходный DataFrame.
            text_col (str): Название колонки с описаниями.
            genre_col (str): Название колонки с жанрами.
            rating_col (str): Название колонки со средним рейтингом.
            num_rating_col (str): Название колонки с количеством рейтингов.

        Returns:
            pd.DataFrame: DataFrame с добавленными очищенными/обработанными колонками.
        """
        print("Начало предварительной обработки DataFrame...")
        # Создаем копию, чтобы не изменять исходный DataFrame
        processed_df = df.copy()

        # 1. Очистка текстовых описаний
        print(f"Очистка колонки '{text_col}'...")
        # Заполняем пропуски пустой строкой перед очисткой
        processed_df[text_col] = processed_df[text_col].fillna('')
        processed_df['cleaned_description'] = processed_df[text_col].apply(clean_text)

        # 2. Парсинг жанров
        print(f"Парсинг колонки '{genre_col}'...")
        processed_df['genre_list'] = processed_df[genre_col].apply(self._parse_genres)

        # 3. Очистка количества рейтингов
        print(f"Очистка колонки '{num_rating_col}'...")
        processed_df['num_ratings_cleaned'] = processed_df[num_rating_col].apply(self._clean_num_ratings)

        # 4. Обработка пропусков в рейтинге (заполним нулем для простоты)
        print(f"Обработка пропусков в '{rating_col}'...")
        processed_df[rating_col] = processed_df[rating_col].fillna(0)

        print("Предварительная обработка завершена.")
        self.processed_df = processed_df # Сохраняем обработанный DataFrame
        return processed_df

    def fit_transform_features(self, processed_df, cleaned_text_col='cleaned_description',
                               genre_list_col='genre_list', rating_col='Avg_Rating',
                               num_ratings_cleaned_col='num_ratings_cleaned'):
        """
        Обучает векторизаторы/скейлеры и преобразует признаки.

        Args:
            processed_df (pd.DataFrame): DataFrame после preprocess_dataframe.
            cleaned_text_col (str): Колонка с очищенными описаниями.
            genre_list_col (str): Колонка со списками жанров.
            rating_col (str): Колонка со средним рейтингом (без пропусков).
            num_ratings_cleaned_col (str): Колонка с очищенным количеством рейтингов.
        """
        if processed_df is None:
            print("Ошибка: DataFrame не был обработан. Запустите preprocess_dataframe() сначала.")
            return

        print("Начало векторизации и нормализации признаков...")

        # 1. Векторизация описаний (TF-IDF)
        print(f"Векторизация '{cleaned_text_col}'...")
        self.description_matrix = self.tfidf_vectorizer.fit_transform(processed_df[cleaned_text_col])
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out() # Сохраняем имена признаков
        print(f"Размер матрицы TF-IDF: {self.description_matrix.shape}")

        # 2. Векторизация жанров (MultiLabelBinarizer)
        print(f"Векторизация '{genre_list_col}'...")
        self.genre_matrix = self.mlb.fit_transform(processed_df[genre_list_col])
        print(f"Размер матрицы жанров: {self.genre_matrix.shape}")

        # 3. Нормализация среднего рейтинга
        print(f"Нормализация '{rating_col}'...")
        # Используем .values.reshape(-1, 1) для преобразования в 2D массив, необходимый скейлеру
        ratings_reshaped = processed_df[rating_col].values.reshape(-1, 1)
        self.normalized_avg_ratings = self.rating_scaler.fit_transform(ratings_reshaped)
        print(f"Размер нормализованных средних рейтингов: {self.normalized_avg_ratings.shape}")

        # 4. Нормализация количества рейтингов
        print(f"Нормализация '{num_ratings_cleaned_col}'...")
        num_ratings_reshaped = processed_df[num_ratings_cleaned_col].values.reshape(-1, 1)
        self.normalized_num_ratings = self.num_rating_scaler.fit_transform(num_ratings_reshaped)
        print(f"Размер нормализованного количества рейтингов: {self.normalized_num_ratings.shape}")

        print("Векторизация и нормализация завершены.")

    def get_combined_features(self, include_genres=True, include_description=True):
        """
        Объединяет выбранные матрицы признаков (TF-IDF описаний и/или OHE жанров).

        Args:
            include_genres (bool): Включать ли матрицу жанров.
            include_description (bool): Включать ли матрицу TF-IDF описаний.

        Returns:
            scipy.sparse.csr_matrix: Объединенная разреженная матрица признаков.
                                      Возвращает None, если нет признаков для объединения.
        """
        features_to_stack = []
        if include_description and self.description_matrix is not None:
            features_to_stack.append(self.description_matrix)
            print("Добавляем матрицу описаний в итоговую.")
        if include_genres and self.genre_matrix is not None:
            features_to_stack.append(self.genre_matrix)
            print("Добавляем матрицу жанров в итоговую.")

        if not features_to_stack:
            print("Нет признаков для объединения.")
            return None

        # hstack требует, чтобы все матрицы имели одинаковое количество строк
        # Объединяем матрицы горизонтально
        combined_matrix = hstack(features_to_stack).tocsr() # tocsr() - эффективный формат для вычислений
        print(f"Размер объединенной матрицы признаков: {combined_matrix.shape}")
        return combined_matrix

# Пример использования (для отладки)
# if __name__ == '__main__':
#     # Создадим пример DataFrame
#     data = {
#         'Description': ["A great book about wizards!", "A classic novel of manners.", None, "Another book description."],
#         'Genres': ["['Fantasy', 'Magic']", "['Classics', 'Romance']", "['Fiction']", None],
#         'Avg_Rating': [4.5, 4.2, 3.9, float('nan')],
#         'Num_Ratings': ["1,234", "5,678", "910", "1,000,000"]
#     }
#     dummy_df = pd.DataFrame(data)
#     print("Исходный DataFrame:")
#     print(dummy_df)
#
#     feature_engineer = FeatureEngineer()
#
#     # Шаг 1: Предобработка
#     processed_df = feature_engineer.preprocess_dataframe(dummy_df)
#     print("\nDataFrame после предобработки:")
#     print(processed_df[['cleaned_description', 'genre_list', 'num_ratings_cleaned', 'Avg_Rating']])
#
#     # Шаг 2: Обучение и трансформация признаков
#     feature_engineer.fit_transform_features(processed_df)
#
#     # Шаг 3: Получение объединенной матрицы
#     combined_features = feature_engineer.get_combined_features(include_genres=True, include_description=True)
#
#     if combined_features is not None:
#         print("\nОбъединенная матрица признаков (первые 5 признаков):")
#         print(combined_features[:, :5].toarray()) # toarray() для наглядности, но матрица разреженная
#         print(f"\nИмена первых 5 признаков TF-IDF: {feature_engineer.feature_names[:5]}")
#         print(f"\nКлассы (жанры) из MultiLabelBinarizer: {feature_engineer.mlb.classes_}")
#