import numpy as np

def get_top_k_similar(sim_scores_enum, k, self_index):
    """
    Находит индексы top-k самых похожих элементов, исключая сам элемент.

    Args:
        sim_scores_enum (list): Список кортежей (index, similarity_score).
        k (int): Количество топ-элементов для возврата.
        self_index (int): Индекс самого элемента, который нужно исключить.

    Returns:
        set: Множество индексов top-k похожих элементов.
    """
    # Сортируем по убыванию схожести
    sorted_scores = sorted(sim_scores_enum, key=lambda x: x[1], reverse=True)

    # Собираем индексы, исключая self_index
    top_k_indices = set()
    count = 0
    for index, score in sorted_scores:
        if index != self_index:
            top_k_indices.add(index)
            count += 1
            if count >= k:
                break
    return top_k_indices

def precision_at_k(recommended_indices, relevant_indices, k):
    """
    Вычисляет Precision@k.

    Args:
        recommended_indices (set): Множество индексов рекомендованных элементов (top-k).
        relevant_indices (set): Множество индексов релевантных элементов (например, top-k по схожести).
        k (int): Количество рекомендаций.

    Returns:
        float: Значение Precision@k.
    """
    if k == 0:
        return 0.0
    # Находим пересечение множеств
    intersection = recommended_indices.intersection(relevant_indices)
    # Точность = |Пересечение| / |Рекомендованные|
    # Так как мы рекомендуем k элементов, знаменатель = k
    return len(intersection) / k

def recall_at_k(recommended_indices, relevant_indices):
    """
    Вычисляет Recall@k.

    Args:
        recommended_indices (set): Множество индексов рекомендованных элементов (top-k).
        relevant_indices (set): Множество индексов релевантных элементов (например, top-k по схожести).

    Returns:
        float: Значение Recall@k.
    """
    if not relevant_indices: # Проверка, если релевантных нет (деление на 0)
        return 0.0
     # Находим пересечение множеств
    intersection = recommended_indices.intersection(relevant_indices)
     # Полнота = |Пересечение| / |Релевантные|
    return len(intersection) / len(relevant_indices)

def f1_at_k(precision, recall):
    """
    Вычисляет F1-score@k.

    Args:
        precision (float): Значение Precision@k.
        recall (float): Значение Recall@k.

    Returns:
        float: Значение F1-score@k.
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

# Пример использования (покажет, как вызывать функции,
# но реальная оценка будет в main.py или отдельном скрипте)
# if __name__ == '__main__':
#     # Пример: есть 5 элементов. Ищем top-3 релевантных/рекомендованных для элемента 0.
#     # Допустим, оценки схожести элемента 0 с другими:
#     sim_scores_enum_example = [(0, 1.0), (1, 0.9), (2, 0.5), (3, 0.8), (4, 0.2)]
#     self_index_example = 0
#     k_example = 3
#
#     # "Релевантные" - топ-3 по схожести (индексы 1, 3, 2)
#     relevant_set = get_top_k_similar(sim_scores_enum_example, k_example, self_index_example)
#     print(f"Релевантные (top-{k_example} по схожести): {relevant_set}") # {1, 2, 3}
#
#     # "Рекомендованные" - допустим, наш алгоритм с учетом рейтинга выдал топ-3: индексы 3, 1, 4
#     recommended_set = {3, 1, 4}
#     print(f"Рекомендованные (top-{k_example} от алгоритма): {recommended_set}")
#
#     # Считаем метрики
#     p_at_k = precision_at_k(recommended_set, relevant_set, k_example)
#     r_at_k = recall_at_k(recommended_set, relevant_set)
#     f1 = f1_at_k(p_at_k, r_at_k)
#
#     print(f"\nPrecision@{k_example}: {p_at_k:.4f}") # Пересечение {1, 3} -> 2 / 3 = 0.6667
#     print(f"Recall@{k_example}: {r_at_k:.4f}")    # Пересечение {1, 3} -> 2 / 3 = 0.6667
#     print(f"F1-score@{k_example}: {f1:.4f}")
#