import csv
import sys
from typing import Dict, List

# ---- Safe CSV field size limit (Windows-compatible) ----
max_size = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_size)
        break
    except OverflowError:
        max_size = max_size // 10


def load_msmarco_corpus(path: str) -> Dict[str, str]:
    corpus = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            doc_id = row[0]   # e.g., D1920946
            text = row[2]
            corpus[doc_id] = text
    return corpus


def load_msmarco_queries(path: str) -> Dict[str, str]:
    queries = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            qid = row[0]
            text = row[1]
            queries[qid] = text
    return queries


def load_msmarco_qrels(path: str) -> Dict[str, List[str]]:
    qrels = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 4:
                continue
            qid = row[0]
            doc_id = row[2]
            rel = int(row[3])
            if rel > 0:
                qrels.setdefault(qid, []).append(doc_id)
    return qrels
