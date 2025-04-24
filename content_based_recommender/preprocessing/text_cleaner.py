import re # Модуль для работы с регулярными выражениями
import string # Модуль для работы со строками (содержит знаки пунктуации)
import nltk # Библиотека для обработки естественного языка
# Убедитесь, что стоп-слова и токенизатор загружены (мы это делали ранее)
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Получаем список английских стоп-слов
# Если описания на русском, замените 'english' на 'russian'
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Выполняет базовую очистку текста:
    1. Приводит к нижнему регистру.
    2. Удаляет знаки пунктуации.
    3. Удаляет числа (опционально, пока оставим).
    4. Токенизирует текст.
    5. Удаляет стоп-слова.
    6. Объединяет токены обратно в строку.

    Args:
        text (str): Исходная строка текста (описание книги).

    Returns:
        str: Очищенная строка текста.
             Возвращает пустую строку, если входной текст некорректен.
    """
    if not isinstance(text, str):
        # Обработка случаев, если в колонке не строка (например, NaN/float)
        return ""

    # 1. Приведение к нижнему регистру
    text = text.lower()

    # 2. Удаление пунктуации
    # Создаем таблицу для перевода: все знаки пунктуации заменяются на None (удаляются)
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # 3. Удаление чисел (раскомментируйте, если нужно удалять цифры)
    # text = re.sub(r'\d+', '', text)

    # 4. Токенизация (разделение на слова)

  
    tokens = text.split()

    # 5. Удаление стоп-слов
    cleaned_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    # isalpha() используется, чтобы убрать возможные остатки пунктуации или пустые строки

    # 6. Объединение токенов обратно в строку
    cleaned_text = ' '.join(cleaned_tokens)

    return cleaned_text

# Пример использования (можно раскомментировать для проверки)
# if __name__ == '__main__':
#     example_desc = "This is Book #1's description, it's really great! Read it 10 times."
#     cleaned_desc = clean_text(example_desc)
#     print(f"Original: {example_desc}")
#     print(f"Cleaned:  {cleaned_desc}")

#     example_desc_ru = "Это описание Книги №1, оно очень хорошее! Читал 10 раз."
#     # Если будете тестировать русский, не забудьте поменять язык стоп-слов выше
#     # stop_words = set(stopwords.words('russian'))
#     # cleaned_desc_ru = clean_text(example_desc_ru)
#     # print(f"Original RU: {example_desc_ru}")
#     # print(f"Cleaned RU:  {cleaned_desc_ru}")

#     print(clean_text(123)) # Проверка на не-строку