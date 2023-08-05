from transformers import pipeline
import traceback

models = [
    "stefan-it/german-gpt2-larger",
    "malteos/bloom-6b4-clp-german",
    "ai-forever/mGPT",
    "DAMO-NLP-MT/polylm-13b",
    "OpenBuddy/openbuddy-llama2-13b-v8.1-fp16",
    "facebook/xglm-564M",
    "facebook/xglm-1.7B",
    "facebook/xglm-2.9B",
    "facebook/xglm-4.5B",
    "facebook/xglm-7.5B"
]

for model in models:
    print(f"Now loading model {model}")
    pipe = pipeline("text-generation", model = model, framework = "pt")
    
    try:
        print(pipe("Ich bin ein Sprachmodell, also"))
    except:
        traceback.print_exec()
    finally:
        del pipe