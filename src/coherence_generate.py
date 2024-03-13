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
    # model name, batch_size, device, device Mapping
    # ("stefan-it/german-gpt2-larger", 64, 0, None),
    # ("ai-forever/mGPT", 4, 0, None),
    # ("facebook/xglm-564M", 16, 0, None),
    # ("facebook/xglm-1.7B", 4, 0, None),
    ("facebook/xglm-2.9B", 16, 0, None), # The larger models should be ran on a bigger GPU 
    ("facebook/xglm-4.5B", 1, 0, None), # The larger models should be ran on a bigger GPU
    ("facebook/xglm-7.5B", 1, 0, None), # The larger models should be ran on a bigger GPU
    # ("malteos/bloom-350m-german", 32, 0, None),
    # ("malteos/bloom-1b5-clp-german", 4, 0, None)
    ("malteos/bloom-6b4-clp-german", 0, 0, None) # The larger models should be ran on a bigger GPU
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

for model_name, batch_size, device, device_map in models:
       
    print(f"now loading: {model_name}")
    model = pipeline("text-generation", model = model_name, device = device, device_map = device_map)
    model.tokenizer.pad_token_id = model.model.config.eos_token_id
    model.tokenizer.padding_side = "left"
    print(model.device)
    
    rows = []
    n = 0

    for np1, np2, female in male_pairing + female_pairing:
        for cat, verb in verb_list:
            try:
                n += 1
                prompt = np1 + " " + verb + " " + np2 + ","
                nrow = {"np1": np1, "np2": np2, "female": female, "cat": cat, "verb": verb, "prompt": prompt}
                rows.append(nrow)
            except Exception as e:
                print(e)
                    
    exp2 = pd.DataFrame(rows, columns = ["con", "np1", "np2", "female", "cat", "verb", "prompt"])
    
    prompts = PromptDataset(exp2["prompt"].tolist())
    conts = []
    
    for out in tqdm(model(prompts, batch_size = batch_size, remove_invalid_values=True, early_stopping = True, do_sample = False, diversity_penalty = .6, num_beam_groups = 10, num_beams = 10, max_new_tokens = 25), total = len(prompts)):
        conts += [model_out["generated_text"] for model_out in out]
    exp2["continuation"] = pd.Series(conts)
    
    exp2.to_csv(f"../data/coherence--{model_name.replace('/', '--')}.csv", sep=";", index=False)
    
    del model
    del exp2
    del conts
    del prompts
    