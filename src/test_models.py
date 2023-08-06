from transformers import pipeline

import traceback

models = [
    "stefan-it/german-gpt2-larger",
    "malteos/bloom-6b4-clp-german",
    "ai-forever/mGPT",
    "facebook/xglm-564M",
    "facebook/xglm-1.7B",
    "facebook/xglm-2.9B",
    "facebook/xglm-4.5B"
]

for model_name in models:
    print(f"Now loading model {model_name}")
    model = pipeline("text-generation", model = model_name)
    print(model("Ich bin ein Sprachmodell, also"))