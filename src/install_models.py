from transformers import pipeline
import traceback

models = ["stefan-it/german-gpt2-larger"]

for model in models:
    pipe = pipeline("text-generation", model = model)
    
    try:
        print(pipe("Ich bin ein Sprachmodell, also"))
    except:
        traceback.print_exec()
    finally:
        del pipe