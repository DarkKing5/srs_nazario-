import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ContentRecommender:
    """
    Класс для генерации рекомендаций книг на основе контента (описания, жанры)
    с учетом рейтинга.
    """
    def __init__(self, dataframe, feature_matrix, normalized_ratings):
        """
        Инициализирует рекомендатель.

        Args:
            dataframe (pd.DataFrame): DataFrame с информацией о книгах
                                      (нужен для получения названий и исходных рейтингов).
                                      Должен содержать колонку 'Book' (или другую с названиями).
            feature_matrix (scipy.sparse.csr_matrix): Объединенная матрица признаков
                                                       (TF-IDF + Жанры).
            normalized_ratings (np.array): 1D или 2D массив нормализованных
                                           рейтингов (Avg_Rating), соответствующий
                                           порядку строк в dataframe и feature_matrix.
        """
        if 'Book' not in dataframe.columns:
             raise ValueError("DataFrame должен содержать колонку 'Book' с названиями книг.")

        self.df = dataframe.reset_index() # Сбрасываем индекс, чтобы можно было легко искать по index
        self.feature_matrix = feature_matrix
        # Убедимся, что normalized_ratings это плоский массив (1D)
        self.normalized_ratings = np.array(normalized_ratings).flatten()

        # Проверка соответствия размеров
        if not (len(self.df) == self.feature_matrix.shape[0] == len(self.normalized_ratings)):
            raise ValueError(f"Размеры DataFrame ({len(self.df)}), "
                             f"матрицы признаков ({self.feature_matrix.shape[0]}) и "
                             f"рейтингов ({len(self.normalized_ratings)}) должны совпадать.")

        print("Создание индекса 'Название -> Индекс'...")
        # Создаем словарь для быстрого поиска индекса книги по названию
        # Используем Series для более эффективного поиска
        self.title_to_index = pd.Series(self.df.index, index=self.df['Book']).to_dict()
        print("Индекс создан.")

        print("Вычисление матрицы косинусного сходства...")
        # Вычисляем косинусное сходство между всеми книгами один раз
        self.cosine_sim = cosine_similarity(self.feature_matrix, self.feature_matrix)
        print(f"Матрица сходства вычислена. Размер: {self.cosine_sim.shape}")

    def recommend(self, book_title, num_recommendations=10, rating_weight=0.3):
        """
        Генерирует рекомендации для заданной книги.

        Args:
            book_title (str): Название книги, для которой ищутся рекомендации.
            num_recommendations (int): Количество рекомендуемых книг.
            rating_weight (float): Вес рейтинга при расчете итоговой оценки (от 0 до 1).
                                   0 - учитывать только сходство, 1 - только рейтинг.

        Returns:
            list: Список названий рекомендованных книг.
                  Возвращает пустой список, если книга не найдена или произошла ошибка.
        """
        print(f"\nПоиск рекомендаций для книги: '{book_title}'")

        # 1. Найти индекс книги по названию
        if book_title not in self.title_to_index:
            print(f"Ошибка: Книга '{book_title}' не найдена в базе данных.")
            # Попробуем найти похожие названия (простая проверка на вхождение)
            possible_matches = [title for title in self.title_to_index.keys() if book_title.lower() in title.lower()]
            if possible_matches:
                print("Возможно, вы имели в виду:")
                for match in possible_matches[:5]: # Показать до 5 совпадений
                    print(f"- {match}")
            return []

        book_index = self.title_to_index[book_title]

        # 2. Получить оценки сходства этой книги со всеми остальными
        # `enumerate` добавляет индекс к каждому элементу сходства
        sim_scores_enum = list(enumerate(self.cosine_sim[book_index]))

        # 3. Рассчитать взвешенную оценку (сходство + рейтинг)
        weighted_scores = []
        for i, similarity in sim_scores_enum:
            # Пропускаем саму книгу
            if i == book_index:
                continue

            # Получаем нормализованный рейтинг для книги i
            normalized_rating = self.normalized_ratings[i]

            # Формула: чем выше вес, тем важнее рейтинг
            combined_score = (1 - rating_weight) * similarity + rating_weight * normalized_rating
            weighted_scores.append((i, combined_score))

        # 4. Отсортировать книги по убыванию взвешенной оценки
        sorted_scores = sorted(weighted_scores, key=lambda x: x[1], reverse=True)

        # 5. Выбрать топ N книг
        top_books_indices = [i for i, score in sorted_scores[:num_recommendations]]

        # 6. Получить названия топ N книг из DataFrame
        recommended_books = self.df.iloc[top_books_indices]['Book'].tolist()

        print(f"Рекомендации найдены ({len(recommended_books)} шт.).")
        return recommended_books

# Пример использования (для отладки или понимания)
# if __name__ == '__main__':
#     # Здесь нужно было бы создать тестовые:
#     # - DataFrame (test_df)
#     # - Матрицу признаков (test_matrix) из TF-IDF + OHE
#     # - Нормализованные рейтинги (test_norm_ratings)
#
#     # --- Начало блока ЗАГЛУШЕК (для примера) ---
#     # В реальном коде эти данные придут из FeatureEngineer
#     data = {
#         'Book': ['Книга A', 'Книга B', 'Книга C', 'Книга D'],
#         'Avg_Rating': [4.0, 3.0, 5.0, 4.5] # Пример исходных рейтингов
#     }
#     test_df = pd.DataFrame(data)
#     # Примерная матрица признаков (4 книги, 5 признаков) - в реальности будет разреженная
#     test_matrix = np.array([
#         [1, 1, 0, 0, 1], # A
#         [1, 1, 0, 1, 0], # B (похожа на A)
#         [0, 0, 1, 1, 1], # C (непохожа на A, B)
#         [1, 0, 0, 0, 1]  # D (похожа на A)
#     ])
#     # Примерные нормализованные рейтинги (от 0 до 1)
#     test_norm_ratings = np.array([0.5, 0.0, 1.0, 0.75]) # (4-3)/(5-3)=0.5, (3-3)/(5-3)=0, (5-3)/(5-3)=1, (4.5-3)/(5-3)=0.75
#     # --- Конец блока ЗАГЛУШЕК ---
#
#     try:
#         recommender = ContentRecommender(test_df, test_matrix, test_norm_ratings)
#
#         # Получаем рекомендации для 'Книга A'
#         recommendations = recommender.recommend('Книга A', num_recommendations=2, rating_weight=0.5)
#         print(f"\nРекомендации для 'Книга A': {recommendations}") # Ожидаем D и B, порядок зависит от рейтинга
#
#         # Попробуем найти несуществующую книгу
#         recommendations_bad = recommender.recommend('Неизвестная Книга')
#
#     except ValueError as e:
#         print(e)