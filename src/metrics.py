def precision_at_k(pred, gt, k):
    pred = pred[:k]
    hits = len(set(pred) & set(gt))
    return hits / k

def recall_at_k(pred, gt, k):
    pred = pred[:k]
    hits = len(set(pred) & set(gt))
    return hits / len(gt)

def f1_at_k(p, r):
    if p + r == 0:
        return 0
    return 2 * p * r / (p + r)
