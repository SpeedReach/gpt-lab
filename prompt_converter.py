import json


with open("data/public_train.jsonl", "r", encoding="utf8") as json_file:
    json_list = list(json_file)

list = [json.loads(json_str) for json_str in json_list]

with open('output/empty_data.jsonl', 'a', encoding='utf8') as json_file:
    for data in list:
        json.dump({"prompt": data["claim"], "completion":" "}, json_file, ensure_ascii=False)
        json_file.write('\n')

with open('output/filled_data.jsonl', 'a', encoding='utf8') as json_file:

    for data in list:
        if data["label"] == "NOT ENOUGH INFO":
            continue
        claim = data["claim"]
        evidences = set()
        for es in data["evidence"]:
            for e in es:    
                term = e[2]
                if (((new_term := term) in claim) or
                    ((new_term := term.split("_")[0]) in claim) or
                    ((new_term := term.replace("·", "")) in claim) or
                    ((new_term := term.split(" ")[0]) in claim) or
                    ((new_term := term.replace("-", " ")) in claim)):
                    matched = True

                elif "·" in term:
                    splitted = term.split("·")
                    for split in splitted:
                        if (new_term := split) in claim:
                            matched = True
                            break
                new_term = new_term.replace(" ", "_")
                new_term = new_term.replace("-", "")
                evidences.add(new_term)
        
        evidence = " " + "\n".join(evidences)
        json.dump({"prompt": data["claim"]+"\n\n###\n\n", "completion":evidence+" END"}, json_file, ensure_ascii=False)
        json_file.write('\n')