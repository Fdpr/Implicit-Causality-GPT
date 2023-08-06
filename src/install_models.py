from transformers import AutoConfig, pipeline, AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import traceback

models = [
    ("stefan-it/german-gpt2-larger",
    "stefan-it--german-gpt2-larger"),
    ("malteos/bloom-6b4-clp-german",
    "malteos--bloom-6b4-clp-german"),
    ("ai-forever/mGPT",
    "ai-forever--mGPT"),
    ("DAMO-NLP-MT/polylm-13b",
    "DAMO-NLP-MT--polylm-13b"),
    ("OpenBuddy/openbuddy-llama2-13b-v8.1-fp16",
    "OpenBuddy--openbuddy-llama2-13b-v8.1-fp16"),
    ("facebook/xglm-564M",
    "facebook--xglm-564M"),
    ("facebook/xglm-1.7B",
    "facebook--xglm-1.7B"),
    ("facebook/xglm-2.9B",
    "facebook--xglm-2.9B"),
    ("facebook/xglm-4.5B",
    "facebook--xglm-4.5B"),
    ("facebook/xglm-7.5B",
    "facebook--xglm-7.5B")
]

for model_name, local_path in models:
    print(f"Now loading model {model_name}")
    
    config = AutoConfig.from_pretrained(model_name)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    model = load_checkpoint_and_dispatch(model, f"~/.cache/huggingface/hub/models--{local_path}", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    try:
        inputs = tokenizer("Hello, my dog is cute and ", return_tensors="pt")
        print(model.generate(**inputs))
    except:
        traceback.print_exec()
    finally:
        del model
        del tokenizer