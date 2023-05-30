from typing import List, Tuple


def calculate_precisions(
        predictions: List[List[str]],
        reference: List[List[str]]
) -> float:
    precision = 0
    count = 0

    for i, d in enumerate(reference):
        if len(d) == 0:
            continue

        predicted_titles = predictions[i]
        precision += calculate_precision(predicted_titles, d)

        count += 1

    return precision / count

def calculate_precision(
        predicted_titles: List[str],
        ground_evidences: List[str]
) -> float:
    hits = 0
    
    for title in predicted_titles:
        if title in ground_evidences:
            hits += 1
    
    return hits / len(predicted_titles)


def calculate_recall(
        predicted_titles: List[str],
        ground_evidences: List[str]
) -> float:
    hits = 0
    
    for title in predicted_titles:
        if title in ground_evidences:
            hits += 1
    
    return hits / len(ground_evidences)

def calculate_recalls(
        predictions: List[List[str]],
        reference: List[List[str]]
) -> float:
    recall = 0
    count = 0

    for i, d in enumerate(reference):
        if len(d) == 0:
            continue

        predicted_titles = predictions[i]
        recall=calculate_recall(predicted_titles, d)

        count += 1

    return recall / count


beta = 0.7