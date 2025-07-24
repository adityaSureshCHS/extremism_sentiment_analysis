# minimal_hatebert.py
# text = "I hate you so much."  # <-- variable storing the text
# text = "All Jews are disgusting and should be removed from our country."
text = "I love my parents so much."
from transformers import pipeline
from huggingface_hub.utils import HFValidationError, RepositoryNotFoundError

PRIMARY  = "GroNLP/hateBERT-hateXplain"
FALLBACK = "Hate-speech-CNERG/bert-base-uncased-hatexplain"

# load model once
try:
    clf = pipeline("text-classification", model=PRIMARY, tokenizer=PRIMARY, top_k=None, truncation=True)
except (HFValidationError, RepositoryNotFoundError, OSError):
    clf = pipeline("text-classification", model=FALLBACK, tokenizer=FALLBACK, top_k=None, truncation=True)

outputs = clf(text)[0]  # list of dicts: {'label': ..., 'score': ...}
hate_prob = next((o["score"] for o in outputs if "hate" in o["label"].lower()), 0.0)

print(f"Hate probability: {hate_prob:.4f}")
print("All scores:", outputs)