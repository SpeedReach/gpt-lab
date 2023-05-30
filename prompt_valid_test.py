import openai
import json
import pandas as pd
import precision_calculate as pc
import wikipedia
import re
from st_correction import do_st_corrections

wikipedia.set_lang("zh")
end_of_prompt = "\n\n###\n\n"
model = "babbage:ft-personal-2023-05-30-07-54-13"
results = open('output/results.jsonl', 'w', encoding='utf8')

precision_count = 0
recall_count = 0
recall = 0
precision = 0

def search_wiki(key_word) -> str:
    r = wikipedia.search(key_word, results=1)
    if len(r) == 0:
        return ""
    r = r[0]
    r = do_st_corrections(r)
    r = re.sub(r"\s\(\S+\)", "", r)
    return r


with open('output/filled_data_prepared_valid.jsonl', 'r', encoding='utf8') as json_file:
    json_list = list(json_file)
    for json_str in json_list[:700]:
        result = json.loads(json_str)
        prompt = result["prompt"]
        evidence_pages = result["completion"].split(" ")[1].split("\n")
        
        result = openai.Completion.create(
            model=model,
            prompt=prompt+end_of_prompt,
            max_tokens=30,
            temperature=0,
            stop=["END"]
            )
        predicted_pages = result["choices"][0]["text"].strip().split("\n")
        json.dump({
            "prompt":prompt,
            "predicted_pages":predicted_pages
        }, results, ensure_ascii=False)
        results.write("\n")
        predicted_pages = list(set(map(search_wiki, predicted_pages)))


        if len(evidence_pages) != 0:
            precision+=pc.calculate_precision(predicted_pages, evidence_pages)
            precision_count += 1
        
        recall+=pc.calculate_recall(predicted_pages, evidence_pages)
        recall_count += 1
print("precision: ", precision/precision_count)
print("recall: ", recall/recall_count)
    