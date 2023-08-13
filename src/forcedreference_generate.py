from transformers import pipeline, AutoTokenizer
import pandas as pd
import json
from random import Random
from torch.utils.data import Dataset
from tqdm import tqdm
import transformers
import traceback
from Annotation import annotate
from copy import deepcopy
import itertools
from itertools import permutations

transformers.logging.set_verbosity_error()

ITEMS_PER_CONDITION = 500

class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, idx):
        return self.prompts[idx]
        
def combine(list1, list2, my_bool):
    unique_combinations = []
    permut = permutations(list_1, len(list_2))
    for comb in permut:
        zipped = zip(comb, list_2)
        unique_combinations.append(list(zipped))
    unique_combinations = [item for row in unique_combinations for item in row]
    return [(name1, name2, my_bool) for (name1, name2) in unique_combinations]

models = [
    ("facebook/xglm-1.7B", 4, 0, None),
    ("stefan-it/german-gpt2-larger", 64, 0, None),
    ("ai-forever/mGPT", 4, 0, None),
    ("facebook/xglm-564M", 16, 0, None),
    # model name, batch_size, device, device Mapping
    # ("malteos/bloom-6b4-clp-german", 1, 0, None), #Bloom is too big
    # ("facebook/xglm-2.9B", 16, -1, None), # The larger models should be ran on a bigger GPU
    # ("facebook/xglm-4.5B", 1, -1, "auto") # The larger models should be ran on a bigger GPU
]

with open("../items/names.json", encoding="utf-8") as nfile:
    namedict = json.load(nfile)
male_names = [name for name in namedict["male"]]
female_names = [name for name in namedict["female"]]
with open("../items/verbs_forced_reference.json", encoding="utf-8") as nfile:
    verbdict = json.load(nfile)
es_verbs = verbdict["es"]
se_verbs = verbdict["se"]

male_pairing = combine(male_names, female_names, False)
female_pairing = combine(female_names, male_names, True)
Random(42).shuffle(male_pairing)
Random(84).shuffle(female_pairing)

conditions = [
    (2,  es_verbs, female_pairing, "NP1"),
    (3,  es_verbs,   male_pairing, "NP1"),
    (6,  se_verbs, female_pairing, "NP1"),
    (7,  se_verbs,   male_pairing, "NP1"),
    (10, es_verbs, female_pairing, "NP2"),
    (11, es_verbs,   male_pairing, "NP2"),
    (14, se_verbs, female_pairing, "NP2"),
    (15, se_verbs,   male_pairing, "NP2")
]

items_per_condition = []
  
for condition, verbs, pairing, forced_reference in conditions:
    rows = []
    for verbdict in verbs:
        verb, filler, verbclass = verbdict["verb"], verbdict["filler"], verbdict["verbclass"]
        for np1, np2, female in pairing:
            prompt = f"{np1} {verb} {np2}{filler}, weil"
            nrow = {"condition": condition, "type": "Experiment", "prompt": prompt, "NP1": np1, "NP2": np2, 
                    "NP1gender": "f" if female else "m", "verb": verb, "verbclass": verbclass, "forced": forced_reference}
            rows.append(nrow)
    Random(168).shuffle(rows)
    items_per_condition.append(rows)

for model_name, batch_size, device, device_map in models:
       
    print(f"now loading: {model_name}")
    model = pipeline("text-generation", model = model_name, device = device, device_map = device_map)
    model.tokenizer.pad_token_id = model.model.config.eos_token_id
    model.tokenizer.padding_side = "left"
    
    male_tokens = list(map(model.tokenizer.encode, [" er", " dieser", " jener", " der"]))
    female_tokens = list(map(model.tokenizer.encode, [" sie", " diese", " jene", " die"]))
    
    data = []

    for condition in items_per_condition:
        items = deepcopy(condition)
        bar = tqdm(total = ITEMS_PER_CONDITION)
        bar.set_description(f"Condition {items[0]['condition']}")
        item_iter = iter(items)
        rows = []
        counter = 0
        while bar.n < ITEMS_PER_CONDITION:
            counter += 1
            try:
                row = next(item_iter)
                if row["forced"] == "NP1":
                    tokenized_name = model.tokenizer.encode(f" {row['NP1']}")
                else:
                    tokenized_name = model.tokenizer.encode(f" {row['NP2']}")
                if (row["NP1gender"] == "m" and row["forced"] == "NP1") or (row["NP1gender"] == "f" and row["forced"] == "NP2"):
                    forced_tokens = male_tokens + [tokenized_name]
                else:
                    forced_tokens = female_tokens + [tokenized_name]
                continuation = model(row["prompt"], force_words_ids = [forced_tokens], remove_invalid_values=True, early_stopping = True, do_sample = False, num_beams = 10, max_new_tokens = 25)[0]["generated_text"]
                row["cont"] = continuation[len(row["prompt"]) + 1:]
                res = annotate(row, True)
                if res["Koreferenz"] == row["forced"]:
                    bar.update(1)
                    row.update(res)
                    rows.append(row)
            except StopIteration:
                print(f"Run out of data in condition {items[0]['condition']}")
                break
        del bar
        print(f"Generated {counter} sentences for condition {items[0]['condition']}")
        data += rows

    exp3 = pd.DataFrame(data, columns = ["condition", "type", "prompt", "cont", "NP1", "NP2", "NP1gender", "verb", "verbclass", "forced", "Koreferenz", "Anaphorische Form"])
    exp3.to_csv(f"../data/forced_coreference--{model_name.replace('/', '--')}.csv", sep=";", index=False)
    
    del model
    del exp3
    del rows
    del items
    
    
    # FIX ROW BUG