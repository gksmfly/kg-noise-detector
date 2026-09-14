"""Load the fine-tuned classifier and run single-pair inference with it —
shared by scripts 09 and 10 (the NYT-FB zero-shot probes), which each
duplicated their own copy of `predict()` and the model-loading lines.
"""
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, BertTokenizerFast

from kg_noise.constants import LABEL_GROUNDED


def load_classifier(model_dir: Path):
    """Load the fine-tuned model + tokenizer in eval mode on the best
    available device. Returns (model, tokenizer, device)."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()
    return model, tokenizer, device


def raw_token_length(tokenizer, triple_text: str, sentence: str) -> int:
    """Untruncated token count for a (triple_text, sentence) pair — used to
    tell whether a sample would be truncated at a given max_length."""
    enc = tokenizer(triple_text, sentence, truncation=False)
    return len(enc["input_ids"])


def predict_pair(model, tokenizer, device, triple_text: str, sentence: str,
                  max_length: int) -> tuple[int, float]:
    """Predict one (triple, sentence) pair. Returns (pred_label, prob_grounded)."""
    enc = tokenizer(
        triple_text, sentence, truncation=True, max_length=max_length,
        padding="max_length", return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    prob = torch.softmax(logits, dim=-1)[0]
    pred = int(prob.argmax().item())
    return pred, float(prob[LABEL_GROUNDED].item())
