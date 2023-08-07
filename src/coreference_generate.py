from transformers import pipeline, AutoTokenizer
import pandas as pd
import json
from random import Random
from torch.utils.data import Dataset
from tqdm import tqdm
import transformers
import traceback

transformers.logging.set_verbosity_error()

class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, idx):
        return self.prompts[idx]


models = [
    ("stefan-it/german-gpt2-larger", 64, None),
    ("malteos/bloom-6b4-clp-german", 1, None),
    ("ai-forever/mGPT", 4, None),
    ("facebook/xglm-564M", 16, None),
    ("facebook/xglm-1.7B", 1, None),
    ("facebook/xglm-2.9B", 1, "auto"),
    ("facebook/xglm-4.5B", 1, "auto")
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

for model_name, batch_size, device_map in models:
       
    print(f"now loading: {model_name}")
    model = pipeline("text-generation", model = model_name, device = 0, device_map = device_map)
    model.tokenizer.pad_token_id = model.model.config.eos_token_id
    model.tokenizer.padding_side = "left"
    print(model.device)
    
    rows = []
    n = 0

    for con in ["weil", "sodass"]:
        if n > 128:
            break
        for np1, np2, female in male_pairing + female_pairing:
            if n > 128:
                    break
            for cat, verb in verb_list:
                if n > 128:
                    break
                try:
                    n += 1
                    prompt = f"{np1} {verb} {np2}, {con}"
                    # continuation = model(prompt, remove_invalid_values=True, early_stopping = True, do_sample = False, diversity_penalty = .8, num_beam_groups = 5, num_beams = 10, max_new_tokens = 15)[0]["generated_text"][len(prompt):]
                    # continuation = model(prompt, do_sample = True, top_k = 0, top_p = .95, max_new_tokens = 20)[0]["generated_text"][len(prompt):]
                    nrow = {"prompt": prompt, "con": con, "np1": np1, "np2": np2, "female": female, "cat": cat, "verb": verb}
                    rows.append(nrow)
                except Exception:
                    traceback.print_exc()
                    
    exp1 = pd.DataFrame(rows, columns = ["con", "np1", "np2", "female", "cat", "verb", "prompt"])
    
    prompts = PromptDataset(exp1["prompt"].tolist())
    conts = []
    
    for out in tqdm(model(prompts, batch_size = batch_size, remove_invalid_values=True, early_stopping = True, do_sample = False, diversity_penalty = .5, num_beam_groups = 5, num_beams = 10, max_new_tokens = 18), total = len(prompts)):
        conts += [model_out["generated_text"] for model_out in out]
    exp1["continuation"] = pd.Series(conts)
    
    exp1.to_csv(f"../data/coreference--{model_name.replace('/', '--')}.csv", sep=";", index=False)
    
    del model
    del exp1
    del conts
    del prompts
    