from transformers import pipeline
from torch.utils.data import Dataset
from tqdm import tqdm

models = [
    # model name, batch_size, device, device Mapping
    # ("stefan-it/german-gpt2-larger", 64, 0, None),
    # ("ai-forever/mGPT", 4, 0, None),
    # ("facebook/xglm-564M", 16, 0, None),
    # ("facebook/xglm-1.7B", 4, 0, None),
    ("facebook/xglm-2.9B", 64, 0, None), # The larger models should be ran on a bigger GPU 
    ("facebook/xglm-4.5B", 32, 0, None), # The larger models should be ran on a bigger GPU
    ("facebook/xglm-7.5B", 16, 0, None), # The larger models should be ran on a bigger GPU
    # ("malteos/bloom-350m-german", 32, 0, None),
    # ("malteos/bloom-1b5-clp-german", 4, 0, None),
    ("malteos/bloom-6b4-clp-german", 16, 0, None) # The larger models should be ran on a bigger GPU
]

class MyDataset(Dataset):
    def __init__(self, i):
        super().__init__()
        self.len = i
        
    def __len__(self):
        return self.len*2

    def __getitem__(self, i):
        return "Peter hasste Maria ganz ohne Absicht, weil"

"""
# just load all the models once 
for model, _, _, _ in models:
    pipe = pipeline("text-generation", model = model, device = 0)
""" 

for model, _, _, _ in models:
    pipe = pipeline("text-generation", model = model, device = 0)
    model.tokenizer.pad_token_id = model.model.config.eos_token_id
    model.tokenizer.padding_side = "left"
    for batch_size in [16, 32, 64, 128, 256, 512]:
        print(f"model {model} with batch size {batch_size}")
        dataset = MyDataset(batch_size)
        for out in tqdm(pipe(dataset, batch_size=batch_size, remove_invalid_values=True, early_stopping = True, do_sample = False, diversity_penalty = .6, num_beam_groups = 10, num_beams = 10, max_new_tokens = 25), total=len(dataset)):
            pass
    