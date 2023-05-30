from typing import Dict
import openai
import json
import pandas as pd
import precision_calculate as pc
import wikipedia
import re
from st_correction import do_st_corrections

wikipedia.set_lang("zh")
end_of_prompt = "\n\n###\n\n"
model = "babbage:ft-personal-2023-05-30-06-36-09"
inputs = [
    "data/public_test_data.jsonl",
    "data/private_test_data.jsonl"
]
outputs = [
    "output/public_test_titles.jsonl",
    "output/private_test_titles.jsonl"
]

def search_wiki(key_word) -> str:
    r = wikipedia.search(key_word, results=1)
    if len(r) == 0:
        return ""
    r = r[0]
    r = do_st_corrections(r)
    r = re.sub(r"\s\(\S+\)", "", r)
    return r

def process_test_file(i: int):
    output = open(outputs[i], 'w', encoding='utf8')
    with open(inputs[i], 'r', encoding='utf8') as json_file:
        json_list = list(json_file)
        for json_str in json_list:
            data = json.loads(json_str)
            id = data["id"]
            prompt = data["claim"]
        result = openai.Completion.create(
            model=model,
            prompt=prompt+end_of_prompt,
            max_tokens=30,
            temperature=0,
            stop=["END"]
            )
        predicted_pages = result["choices"][0]["text"].strip().split("\n")
        predicted_pages = list(set(map(search_wiki, predicted_pages)))
        json.dump({
            "id":id,
            "claim":prompt,
            "predicted_pages":predicted_pages
            },
            output, ensure_ascii=False)
        output.write("\n")

process_test_file(0)
process_test_file(1)
