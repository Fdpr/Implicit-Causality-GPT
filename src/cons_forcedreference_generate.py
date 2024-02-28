from transformers import pipeline, AutoTokenizer
import pandas as pd
import json
from random import Random
from torch.utils.data import Dataset
from tqdm import tqdm
import transformers
import traceback
from Annotation_forced import do_annotation
from copy import deepcopy
import itertools
from itertools import product

transformers.logging.set_verbosity_error()

ITEMS_PER_CONDITION = 1000

class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, idx):
        return self.prompts[idx]
        
models = [
    # model name, batch_size, device, device Mapping, is_sentencepiece
    ("stefan-it/german-gpt2-larger", 64, 0, None, False),
    ("ai-forever/mGPT", 4, 0, None, False),
    ("facebook/xglm-564M", 32, 0, None, True),
    ("facebook/xglm-1.7B", 8, 0, None, True),
    # ("facebook/xglm-2.9B", 16, -1, None, True), # The larger models should be ran on a bigger GPU 
    # ("facebook/xglm-4.5B", 1, -1, None, True), # The larger models should be ran on a bigger GPU
    # ("facebook/xglm-7.5B", 1, -1, None, True), # The larger models should be ran on a bigger GPU
    ("malteos/bloom-350m-german", 32, 0, None, False),
    ("malteos/bloom-1b5-clp-german", 4, 0, None, False)
    # ("malteos/bloom-6b4-clp-german", 1, 0, None, False) # The larger models should be ran on a bigger GPU
]

with open("../items/names.json", encoding="utf-8") as nfile:
    namedict = json.load(nfile)
male_names = [name for name in namedict["male"]]
female_names = [name for name in namedict["female"]]
with open("../items/verbs_forced_reference.json", encoding="utf-8") as nfile:
    verbdict = json.load(nfile)
es_verbs = verbdict["es"]
se_verbs = verbdict["se"]

male_pairing = list(product(male_names, female_names, [False]))
female_pairing = list(product(female_names, male_names, [True]))
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

def make_constraint_function(tokenizer, female, is_sentencepiece):
    weil = tokenizer.encode(", sodass")[-2:]
    names = female_names if female else male_names
    if is_sentencepiece:
        pronouns = ["sie", "diese", "jense"] if female else ["er", "dieser", "jener"]
        tokens = list(map(tokenizer.encode, names + pronouns))
        tokens = [toks[1:] for toks in tokens]
    else:
        pronouns = [" sie", " diese", " jense"] if female else [" er", " dieser", " jener"]
        tokens = list(map(tokenizer.encode, list(map(lambda name: " " + name, names)) + pronouns))
    twos = [items for items in tokens if len(items) > 1]
    twos_one = [item[0] for item in twos]
    twos_two = [item[1] for item in twos]
    all_tokens = list(range(tokenizer.vocab_size))
    tokens = [item[0] for item in tokens]
    def constrainer(batch_id, input_tokens):
        if (input_tokens[-2] == weil[-2]) and (input_tokens[-1] == weil[-1]):
            return tokens
        elif (input_tokens[-3] == weil[-2]) and (input_tokens[-2] == weil[-1]) and (input_tokens[-1] in twos_one):
            return [twos_two[twos_one.index(input_tokens[-1])]]
        else:
            return all_tokens
    return constrainer
  
for condition, verbs, pairing, forced_reference in conditions:
    rows = []
    for verbdict in verbs:
        verb, filler, verbclass = verbdict["verb"], verbdict["filler"], verbdict["verbclass"]
        for np1, np2, female in pairing:
            prompt = f"{np1} {verb} {np2}{filler}, sodass"
            nrow = {"condition": condition, "type": "Experiment", "prompt": prompt, "NP1": np1, "NP2": np2, 
                    "NP1gender": "f" if female else "m", "verb": verb, "verbclass": verbclass, "forced": forced_reference}
            rows.append(nrow)
    Random(168).shuffle(rows)
    items_per_condition.append(rows)

for model_name, batch_size, device, device_map, is_sentencepiece in models:
    print(f"now loading: {model_name}")
    model = pipeline("text-generation", model = model_name)
    model.tokenizer.pad_token_id = model.model.config.eos_token_id
    model.tokenizer.padding_side = "left"    
    data = []
    for condition in items_per_condition:
        items = deepcopy(condition)
        constraint_function = make_constraint_function(model.tokenizer, False, is_sentencepiece)
        if ((condition[0]["forced"] == "NP1") and (condition[0]["NP1gender"] == "f")) or ((condition[0]["forced"] == "NP2") and (condition[0]["NP1gender"] == "m")):
            constraint_function = make_constraint_function(model.tokenizer, True, is_sentencepiece)
        bar = tqdm(total = ITEMS_PER_CONDITION)
        bar.set_description(f"Condition {items[0]['condition']}")
        result = []
        counter = 0
        while bar.n < ITEMS_PER_CONDITION:
            counter += batch_size
            if len(items) >= batch_size:
                rows = pd.DataFrame(items[:batch_size])
                items = items[batch_size:]
                prompts = rows["prompt"].to_list()
                continuations = model(prompts, batch_size=batch_size, prefix_allowed_tokens_fn=constraint_function, remove_invalid_values=True, early_stopping = True, do_sample = False, diversity_penalty = .6, num_beam_groups = 10, num_beams = 10, max_new_tokens = 20)
                continuations = [cont[0]["generated_text"] for cont in continuations]
                continuations = list(map(lambda zipped: zipped[1][len(zipped[0])+1:], zip(prompts, continuations)))
                rows["cont"] = continuations
                res = do_annotation(rows, True)
                res = res[res["Koreferenz"] == res["forced"]]
                result += res.values.tolist()
                bar.update(len(res))
            else:
                print(f"Run out of data in condition {condition[0]['condition']}")
                break
        del bar
        print(f"Generated {counter} sentences for condition {items[0]['condition']}")
        data += result

    exp3 = pd.DataFrame(data, columns = ["condition", "type", "prompt", "NP1", "NP2", "NP1gender", "verb", "verbclass", "forced", "cont", "Koreferenz", "Anaphorische Form"])
    exp3.to_csv(f"../data/cons_forced_coreference--{model_name.replace('/', '--')}.csv", sep=";", index=False)
    
    del model
    del exp3
    del rows
    del items
    
    # FIX ROW BUG