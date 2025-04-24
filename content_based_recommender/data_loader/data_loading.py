import pandas as pd
import os

class BookDataLoader:
    """
    Класс для загрузки данных о книгах из CSV файла.
    """
    def __init__(self, file_path):
        """
        Инициализирует загрузчик данных.

        Args:
            file_path (str): Относительный или абсолютный путь к CSV файлу.
        """
        # Проверяем, существует ли файл по указанному пути
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Ошибка: Файл не найден по пути '{file_path}'")
        self.file_path = file_path
        print(f"DataLoader инициализирован с путем: {self.file_path}") # Для отладки

    def load_data(self):
        """
        Загружает данные из CSV файла с использованием pandas.

        Использует первую колонку как индекс.

        Returns:
            pandas.DataFrame: Загруженный DataFrame с данными о книгах.
                             Возвращает None в случае ошибки чтения.
        """
        try:
            # Используем index_col=0, чтобы первая колонка стала индексом DataFrame
            df = pd.read_csv(self.file_path, index_col=0)
            print(f"Данные успешно загружены из {self.file_path}")
            print(f"Колонки в DataFrame: {df.columns.tolist()}") # Посмотрим на колонки
            print(f"Первые 5 строк DataFrame:\n{df.head()}") # Посмотрим на данные
            return df
        except FileNotFoundError:
            # Эта ошибка уже обработана в __init__, но оставим на всякий случай
            print(f"Ошибка: Файл не найден при попытке чтения '{self.file_path}'")
            return None
        except pd.errors.EmptyDataError:
            print(f"Ошибка: Файл '{self.file_path}' пуст.")
            return None
        except Exception as e:
            # Ловим другие возможные ошибки при чтении CSV (например, ошибки парсинга)
            print(f"Произошла ошибка при чтении файла '{self.file_path}': {e}")
            return None

# Пример использования (можно добавить в конец файла для быстрой проверки,
# но потом удалить или закомментировать перед использованием в main.py)
# if __name__ == '__main__':
#     # Укажите ПРАВИЛЬНОЕ имя вашего файла!
#     # Путь '../data/' предполагает, что этот скрипт запускается из папки,
#     # где находится папка content_based_recommender, или что main.py
#     # будет правильно указывать путь при создании DataLoader.
#     # Для простоты проверки можно указать полный путь к файлу.
#     # ЗАМЕНИТЕ 'YOUR_DATA_FILE.csv' НА НАЗВАНИЕ ВАШЕГО ФАЙЛА
#     data_file = 'data/goodreads_data.csv'
#     # Или если запускаете loader.py напрямую:
#     # data_file = '../../data/YOUR_DATA_FILE.csv'
#
#     try:
#         loader = DataLoader(data_file)
#         dataframe = loader.load_data()
#         if dataframe is not None:
#             print("\nПроверка загрузки данных:")
#             print(f"Размер DataFrame: {dataframe.shape}")
#             # print(dataframe.info()) # Раскомментируйте для подробной информации
#     except FileNotFoundError as e:
#         print(e)