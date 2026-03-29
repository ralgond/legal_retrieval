import numpy as np
from collections.abc import Sequence

def citation_f1(
    predicted: Sequence[str],
    gold: Sequence[str],
) -> dict[str, float]:
    """Compute F1 score for citation overlap on a single query.

    Args:
        predicted: List of predicted canonical citation IDs
        gold: List of ground truth canonical citation IDs

    Returns:
        Dictionary with precision, recall, and F1
    """
    pred_set = set(predicted)
    gold_set = set(gold)

    # Edge case: both empty
    if len(pred_set) == 0 and len(gold_set) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    # Edge case: prediction empty but gold not
    if len(pred_set) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Edge case: gold empty but prediction not
    if len(gold_set) == 0:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}

    true_positives = len(pred_set & gold_set)
    precision = true_positives / len(pred_set)
    recall = true_positives / len(gold_set)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}
    
def macro_f1(
    predictions: Sequence[Sequence[str]],
    gold: Sequence[Sequence[str]],
) -> dict[str, float]:
    """Compute Macro F1: average F1 across all queries.

    This is the PRIMARY competition metric.

    Args:
        predictions: List of predicted citation lists (one per query)
        gold: List of gold citation lists (one per query)

    Returns:
        Dictionary with macro precision, recall, and F1
    """
    if len(predictions) != len(gold):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(gold)} gold")

    if len(predictions) == 0:
        return {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}

    precision_scores = []
    recall_scores = []
    f1_scores = []

    for pred, g in zip(predictions, gold):
        scores = citation_f1(pred, g)
        precision_scores.append(scores["precision"])
        recall_scores.append(scores["recall"])
        f1_scores.append(scores["f1"])

    n = len(f1_scores)
    return {
        "macro_precision": sum(precision_scores) / n,
        "macro_recall": sum(recall_scores) / n,
        "macro_f1": sum(f1_scores) / n,
    }
    
def cal_recall(all_hits_l, gold_citations_l, truncate_method=None):
    recalls = []
    for all_hits, gold_citations in zip(all_hits_l, gold_citations_l):   
        all_citation = []
        if truncate_method is None:
            for hit in all_hits:
                all_citation.append(hit)
        else:
            __limit = truncate_method(all_hits)
            for hit in all_hits[:__limit]:
                all_citation.append(hit)

        # print("==>",all_citation[0])
        hits = len(set(all_citation) & set(gold_citations))

        # print("gold_citations.len:", len(gold_citations), ", hits.len:", hits)
        recall = hits / len(gold_citations)
        recalls.append(recall)
        
    mean_recall = np.mean(recalls)
    return mean_recall

def cal_precision(all_hits_l, gold_citations_l, truncate_method=None):
    precisions = []
    for all_hits, gold_citations in zip(all_hits_l, gold_citations_l):
        all_citation = []
        if truncate_method is None:
            for hit in all_hits:
                all_citation.append(hit)
        else:
            __limit = truncate_method(all_hits)
            for hit in all_hits[:__limit]:
                all_citation.append(hit)
        
        predicted = set(all_citation)
        hits = len(predicted & set(gold_citations))
        
        if len(predicted) == 0:
            precision = 0.0
        else:
            precision = hits / len(predicted)
        
        precisions.append(precision)
        
    mean_precision = np.mean(precisions)
    return mean_precision

def cal_f1(r, p):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0