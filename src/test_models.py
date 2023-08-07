from transformers import pipeline, AutoTokenizer

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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    force_flexible = [" er"]
    try:
        force_words_ids = [
            tokenizer(force_flexible, add_prefix_space=True, add_special_tokens=False).input_ids,
        ]
    except TypeError:
        print("no spaces in front")
        force_words_ids = [
            tokenizer(force_flexible, add_special_tokens=False).input_ids,
        ]
    model = pipeline("text-generation", model = model_name)
    # print(model("Ich bin ein Sprachmodell, also", remove_invalid_values=True, early_stopping = True, do_sample = False, diversity_penalty = .6, num_beam_groups = 5, num_beams = 10, max_new_tokens = 20))
    print(model("Andra überraschte Malte, weil", remove_invalid_values=True, num_return_sequences = 1, no_repeat_ngrams = 2, early_stopping = True, force_words_ids=force_words_ids, do_sample = False, num_beams = 10, max_new_tokens = 20))
    del model
    del tokenizer