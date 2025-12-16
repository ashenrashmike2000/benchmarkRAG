import numpy as np
import math

def precision_at_k(pred, gt, k):
    pred = pred[:k]
    return len(set(pred) & set(gt)) / k

def recall_at_k(pred, gt, k):
    pred = pred[:k]
    return len(set(pred) & set(gt)) / max(len(gt), 1)

def reciprocal_rank(pred, gt):
    for i, p in enumerate(pred, start=1):
        if p in gt:
            return 1.0 / i
    return 0.0

def dcg(pred, gt, k):
    score = 0.0
    for i, p in enumerate(pred[:k], start=1):
        if p in gt:
            score += 1.0 / math.log2(i + 1)
    return score

def ndcg_at_k(pred, gt, k):
    ideal = dcg(list(gt), gt, k)
    return dcg(pred, gt, k) / ideal if ideal > 0 else 0.0

def latency_percentiles(latencies):
    return {
        "p50": float(np.percentile(latencies, 50)),
        "p90": float(np.percentile(latencies, 90)),
        "p99": float(np.percentile(latencies, 99)),
    }
def success_at_k(pred, gt, k):
    return 1.0 if any(p in gt for p in pred[:k]) else 0.0


def ndcg_at_k(pred, gt, k):
    def dcg(items):
        score = 0.0
        for i, item in enumerate(items, start=1):
            if item in gt:
                score += 1.0 / math.log2(i + 1)
        return score

    ideal = dcg(list(gt)[:k])
    return dcg(pred[:k]) / ideal if ideal > 0 else 0.0
