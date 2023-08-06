from transformers import AutoConfig, pipeline, AutoTokenizer, LlamaTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from pathlib import Path

HOME = str(Path.home())

import traceback

models = [
    ("stefan-it/german-gpt2-larger",
    "/.cache/huggingface/hub/models--stefan-it--german-gpt2-larger/snapshots/aa2138bb716507181c1bbd288a1076837ed0ca3b/pytorch_model.bin",
    AutoTokenizer,
    True
    ),
    ("malteos/bloom-6b4-clp-german",
    "/.cache/huggingface/hub/models--malteos--bloom-6b4-clp-german/snapshots/2e5a85761d8a39ec602ada390e87ff8d5d316191/pytorch_model.bin.index.json",
    AutoTokenizer,
    False),
    ("ai-forever/mGPT",
    "/.cache/huggingface/hub/models--ai-forever--mGPT/snapshots/cb99dd42d3486c2ed9c14f92ed22e9bc7bbc6ac4/pytorch_model.bin",
    AutoTokenizer,
    True),
    ("DAMO-NLP-MT/polylm-13b",
    "/.cache/huggingface/hub/models--DAMO-NLP-MT--polylm-13b/snapshots/685e5014840342bfdecd55ea961d435357c16a22/pytorch_model.bin.index.json",
    LlamaTokenizer,
    True),
    ("OpenBuddy/openbuddy-llama2-13b-v8.1-fp16",
    "/.cache/huggingface/hub/models--OpenBuddy--openbuddy-llama2-13b-v8.1-fp16/snapshots/c1a5f8fd23e3823ced6efc0ea1df973da850ba7e/pytorch_model.bin.index.json",
    AutoTokenizer,
    True),
    ("facebook/xglm-564M",
    "/.cache/huggingface/hub/models--facebook--xglm-564M/snapshots/f3059f01b98ccc877c673149e0178c0e957660f9/pytorch_model.bin",
    AutoTokenizer,
    True),
    ("facebook/xglm-1.7B",
    "/.cache/huggingface/hub/models--facebook--xglm-1.7B/snapshots/d23a5e8e2164af31a84a26756b9b17f925143050/pytorch_model.bin",
    AutoTokenizer,
    True),
    ("facebook/xglm-2.9B",
    "/.cache/huggingface/hub/models--facebook--xglm-2.9B/snapshots/33c659ae27de09c0a54123d3902dac48cbb8592a/pytorch_model.bin",
    AutoTokenizer,
    True),
    ("facebook/xglm-4.5B",
    "/.cache/huggingface/hub/models--facebook--xglm-4.5B/snapshots/dc6a67fac06c8bca7860b84656a0cb736293a7a8/pytorch_model.bin",
    AutoTokenizer,
    True),
    ("facebook/xglm-7.5B",
    "/.cache/huggingface/hub/models--facebook--xglm-7.5B/snapshots/732d59308a844004bd9a4def972cc7c3896a38e0/pytorch_model.bin",
    AutoTokenizer,
    True)
]

for model_name, local_path, tokenizer, use_accelerate in models:
    print(f"Now loading model {model_name}")
    
    if use_accelerate:  
        config = AutoConfig.from_pretrained(model_name)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
        model = load_checkpoint_and_dispatch(model, HOME + local_path, device_map="auto", offload_folder = ".temp")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = tokenizer.from_pretrained(model_name)
    
    try:
        inputs = tokenizer("Ich bin ein Sprachmodell, also", return_tensors="pt")
        print(tokenizer.decode(model.generate(**inputs)[0]))
    except:
        traceback.print_exc()
    finally:
        del model
        del tokenizer