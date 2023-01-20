import json

def load_jsonl(path) :
    with open(path, "r") as f :
        json_l = [json.loads(l.strip()) for l in f.readlines()]
        f.close()
    return json_l

