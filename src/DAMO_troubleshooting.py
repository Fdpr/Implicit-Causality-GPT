from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch#

from pathlib import Path
HOME = str(Path.home())

model_name = "DAMO-NLP-MT/polylm-13b"
local_path = "/.cache/huggingface/hub/models--DAMO-NLP-MT--polylm-13b/snapshots/685e5014840342bfdecd55ea961d435357c16a22/pytorch_model.bin.index.json"


config = AutoConfig.from_pretrained("DAMO-NLP-MT/polylm-13b")
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
model.tie_weights()
model = load_checkpoint_and_dispatch(model, HOME + local_path, device_map="auto", offload_folder = ".temp", no_split_module_classes=['PolyLMBlock'])
model.eval()
tokenizer = AutoTokenizer.from_pretrained("DAMO-NLP-MT/polylm-13b", legacy=False, use_fast=False)

input_doc = f"Beijing is the capital of China.\nTranslate this sentence from English to Chinese."

inputs = tokenizer(input_doc, return_tensors="pt")

generate_ids = model.generate(
  inputs.input_ids,
  attention_mask=inputs.attention_mask,
  do_sample=False,
  num_beams=4,
  max_length=128,
  early_stopping=True
)

decoded = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(f">>> {decoded}")

