from transformers import pipeline, AutoTokenizer
import pandas as pd
import json
from random import Random
from tqdm import tqdm
import transformers

transformers.logging.set_verbosity_error()

models = [
    "stefan-it/german-gpt2-larger",
    "malteos/bloom-6b4-clp-german",
    "ai-forever/mGPT",
    "facebook/xglm-564M",
    "facebook/xglm-1.7B",
    "facebook/xglm-2.9B",
    "facebook/xglm-4.5B"
]

with open("../items/names.json", encoding="utf-8") as nfile:
    namedict = json.load(nfile)
male_names = [name for name in namedict["male"]]
female_names = [name for name in namedict["female"]]
with open("../items/verbs.json", encoding="utf-8") as vfile:
    verbdict = json.load(vfile)
verbs = {"stim-exp": verbdict["test"]["stim-exp"] + verbdict["train"]["stim-exp"],
         "exp-stim": verbdict["test"]["exp-stim"] + verbdict["train"]["exp-stim"]}


female_shuffled = female_names.copy()
Random(42).shuffle(female_shuffled)
male_shuffled = male_names.copy()
Random(84).shuffle(male_shuffled)
male_pairing = list(zip(male_names, female_shuffled, [False for name in male_names]))
female_pairing = list(zip(female_names, male_shuffled, [True for name in male_names]))
verb_list = [(cat, verb) for cat in verbs.keys() for verb in verbs[cat]]

for model_name in models:

    model = pipeline("text-generation", model = model_name)
    exp1 = pd.DataFrame(columns = ["con", "np1", "np2", "female", "cat", "verb", "continuation"])
    
    bar = tqdm(total = 2 * len(male_pairing) * 2 * len(verb_list))

    for con in ["weil", "sodass"]:
        if bar.n > 10:
            break
        for np1, np2, female in male_pairing + female_pairing:
            if bar.n > 10:
                break
            for cat, verb in verb_list:
                if bar.n > 10:
                break
                try:
                    bar.update(1)
                    prompt = f"{np1} {verb} {np2}, {con}"
                    continuation = model(prompt, remove_invalid_values=True, early_stopping = True, 
                                         do_sample = False, diversity_penalty = .6, num_beam_groups = 5, 
                                         num_beams = 10, max_new_tokens = 20)["generated_text"][len(prompt):]
                    nrow = {"con": con, "np1": np1, "np2": np2, "female": female, "cat": cat, "verb": verb, "continuation": continuation}
                    exp1 = exp1.append(nrow, ignore_index=True)
                except Exception as e:
                    print(e)
    exp1.to_csv(f"../data/coreference{model_name.replace('/', '--'}.csv", sep=";", index=False)
    del model
    del exp1
    