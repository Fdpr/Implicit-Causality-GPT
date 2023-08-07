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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    force_flexible = ["Computer", "Technologie", "Linguistik"] 
    force_words_ids = [
        tokenizer(force_flexible, add_prefix_space=True, add_special_tokens=False).input_ids,
    ]
    model = pipeline("text-generation", model = model_name)
    print(model("Ich bin ein Sprachmodell, also"), do_sample = False, diversity_penalty = .6, num_beam_groups = 5, num_beams = 10, max_new_tokens = 20)