import json

def load_benchmark_queries(path):
    queries = []
    answers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            queries.append(item["query"])
            answers.append(item["ground_truth"])
    return queries, answers

