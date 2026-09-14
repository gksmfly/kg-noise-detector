"""
Step 5: bert-base-cased fine-tuning.

label: positive(corruption_type=none) -> 1 (grounded), negative(entity/relation
corruption) -> 0 (ungrounded/noisy). Validation F1(macro) 기준 best checkpoint
저장, early stopping patience=2.

이 장비에 GPU가 2개 있어 Trainer가 기본적으로 둘 다 묶어 DataParallel로 실행하려다
NCCL 오류로 죽는 문제를 이전 프로젝트에서 겪었다. 그래서 torch를 import하기 전에
CUDA_VISIBLE_DEVICES를 GPU 0 하나로 고정한다.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import json

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    BertTokenizerFast,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from kg_noise.constants import LABEL_GROUNDED, LABEL_NOISY, MODEL_NAME
from kg_noise.io_utils import load_jsonl
from kg_noise.metrics import classification_metrics
from kg_noise.paths import ERRORS_DIR, MODEL_OUT_DIR, SPLITS_DIR

MODEL_OUT = MODEL_OUT_DIR


def load_split(name: str) -> list[dict]:
    return load_jsonl(SPLITS_DIR / f"{name}.jsonl")


def to_hf_dataset(rows: list[dict], tokenizer, max_length: int) -> Dataset:
    triple = [r["triple_text"] for r in rows]
    sentence = [r["sentence"] for r in rows]
    labels = [r["label"] for r in rows]
    enc = tokenizer(
        triple, sentence, padding="max_length", truncation=True,
        max_length=max_length, return_tensors=None,
    )
    ds = Dataset.from_dict({**enc, "labels": labels})
    return ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return classification_metrics(labels, preds)


def main():
    rec_path = SPLITS_DIR / "max_length_recommendation.json"
    max_length = 128
    if rec_path.exists():
        rec = json.loads(rec_path.read_text())
        max_length = rec["recommended_max_length"]
        print(f"[5] Step4 결과에서 max_length={max_length} 사용 (p95={rec['p95_raw']})")
    else:
        print(f"[5] max_length_recommendation.json 없음 -> 기본값 {max_length} 사용")

    print(f"[5] CUDA available: {torch.cuda.is_available()}, "
          f"device count: {torch.cuda.device_count()}")

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    train_rows = load_split("train")
    val_rows = load_split("val")
    test_rows = load_split("test")
    print(f"[5] train {len(train_rows)} / val {len(val_rows)} / test {len(test_rows)}")

    train_ds = to_hf_dataset(train_rows, tokenizer, max_length)
    val_ds = to_hf_dataset(val_rows, tokenizer, max_length)
    test_ds = to_hf_dataset(test_rows, tokenizer, max_length)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    args = TrainingArguments(
        output_dir=str(MODEL_OUT / "checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=20,
        report_to=[],
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    print("[5] === Validation 최종 성능 ===")
    val_metrics = trainer.evaluate(val_ds)
    print(val_metrics)

    print("[5] === Test 성능 ===")
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    print(test_metrics)

    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_OUT / "best"))
    tokenizer.save_pretrained(str(MODEL_OUT / "best"))

    # Step 7/8용: test set 예측 확률 + 정답 + corruption_type 저장
    preds_output = trainer.predict(test_ds)
    probs = torch.softmax(torch.tensor(preds_output.predictions), dim=-1).numpy()
    pred_labels = probs.argmax(axis=-1)

    out_path = ERRORS_DIR / "bert_test_predictions.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r, prob, pred in zip(test_rows, probs, pred_labels):
            f.write(json.dumps({
                "pair_id": r["pair_id"],
                "triple_text": r["triple_text"],
                "sentence": r["sentence"],
                "category": r["category"],
                "corruption_type": r["corruption_type"],
                "true_label": r["label"],
                "pred_label": int(pred),
                "prob_noisy": float(prob[LABEL_NOISY]),
                "prob_grounded": float(prob[LABEL_GROUNDED]),
            }, ensure_ascii=False) + "\n")
    print(f"[5] test 예측 저장 -> {out_path}")

    metrics_path = ERRORS_DIR / "bert_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"val": val_metrics, "test": test_metrics, "max_length": max_length}, f,
                   ensure_ascii=False, indent=2)
    print(f"[5] 메트릭 저장 -> {metrics_path}")


if __name__ == "__main__":
    main()
