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
    ("malteos/bloom-6b4-clp-german", 1, "cpu", None)
]


df_regular = pd.read_csv("../data/Judge_1_not_corrected_ChatGPT.csv", sep=";", index_col=None)
df_durch = pd.read_csv("../data/Judge_2_not_corrected_ChatGPT.csv", sep=";", index_col=None)


for model_name, batch_size, device, device_map in models:
       
    print(f"now loading: {model_name}")
    model = pipeline("text-generation", model = model_name, device = device, device_map = device_map)
    model.tokenizer.pad_token_id = model.model.config.eos_token_id
    model.tokenizer.padding_side = "left"
    print(model.device)
    
    rows = []
    n = 0
    
    prompts_regular = PromptDataset([prompt[:-4] for prompt in df_regular["prompt"].tolist()])
    conts_regular = []
    
    for out in tqdm(model(prompts_regular, batch_size = batch_size, remove_invalid_values=True, early_stopping = True, do_sample = False, diversity_penalty = .6, num_beam_groups = 10, num_beams = 10, max_new_tokens = 25), total = len(prompts_regular)):
        conts_regular += [model_out["generated_text"] for model_out in out]
    df_regular["Bloom"] = conts_regular
    df_regular.to_csv("../data/Judge_1_not_corrected_ChatGPT_Bloom.csv", index=None, sep=";")
    
    prompts_durch = PromptDataset([prompt[:-1] for prompt in df_durch["prompt"].tolist()])
    conts_durch = []
    
    for out in tqdm(model(prompts_durch, batch_size = batch_size, remove_invalid_values=True, early_stopping = True, do_sample = False, diversity_penalty = .6, num_beam_groups = 10, num_beams = 10, max_new_tokens = 25), total = len(prompts_durch)):
        conts_durch += [model_out["generated_text"] for model_out in out]
    df_durch["Bloom"] = conts_durch    
    df_durch.to_csv("../data/Judge_2_not_corrected_ChatGPT_Bloom.csv", index=None, sep=";")
   