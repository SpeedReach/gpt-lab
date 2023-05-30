import json
import precision_calculate as pc

precision_count = 0
precision = 0
recall_count = 0
recall = 0
with open("output/train_doc5.jsonl", "r", encoding="utf8") as json_file:
    json_list = list(json_file)
    for json_str in json_list:
        result = json.loads(json_str)
        if result["label"] == "NOT ENOUGH INFO":
            continue
        evidence = result["evidence"]
        predicted_pages = result["predicted_pages"]
        evidence_title = set()
        for es in evidence:
            for e in es:
                evidence_title.add(e[2])
        if len(evidence_title) != 0:
            precision+=pc.calculate_precision(predicted_pages, evidence_title)
            precision_count += 1
        
        recall+=pc.calculate_recall(predicted_pages, evidence_title)
        recall_count += 1
        
print("precision: ", precision/precision_count)
print("recall: ", recall/recall_count)