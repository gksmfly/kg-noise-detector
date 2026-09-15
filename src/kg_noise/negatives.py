"""Synthetic negative generation shared by every domain's negative-generation
script (originally only `scripts/02_generate_negatives.py` for WebNLG; now
also `scripts/12_generate_negatives_dart.py` for the DART WikiSQL/
WikiTableText probe).

Generates, per positive (subject, relation, object, sentence) row, one
entity_corruption and one relation_corruption negative — same algorithm
regardless of domain, so this is the one place it should live.
"""
import random


def generate_negatives(rows: list[dict], seed: int) -> list[dict]:
    """rows: positive pairs, each with pair_id, category, subject, relation,
    object, triple_text, sentence. Returns a shuffled list of
    positive + entity_corruption + relation_corruption rows (3x len(rows))."""
    rng = random.Random(seed)

    subject_pool = sorted(set(r["subject"] for r in rows))
    object_pool = sorted(set(r["object"] for r in rows))
    relation_pool = sorted(set(r["relation"] for r in rows))

    def other(pool: list[str], current: str) -> str:
        choice = current
        # 아주 드물게 같은 값이 뽑히는 경우를 대비한 재시도 (풀이 1개뿐인 극단적 경우 방지)
        for _ in range(10):
            choice = rng.choice(pool)
            if choice != current:
                break
        return choice

    full = []
    for r in rows:
        base = {"pair_id": r["pair_id"], "category": r["category"]}

        # positive
        full.append({
            **base,
            "triple_text": r["triple_text"],
            "sentence": r["sentence"],
            "subject": r["subject"], "relation": r["relation"], "object": r["object"],
            "corruption_type": "none",
            "label": 1,
        })

        # entity_corruption: subject 또는 object 중 하나를 50/50으로 골라 치환
        if rng.random() < 0.5:
            new_subject = other(subject_pool, r["subject"])
            new_object = r["object"]
            corrupted_slot = "subject"
        else:
            new_subject = r["subject"]
            new_object = other(object_pool, r["object"])
            corrupted_slot = "object"
        entity_triple = f"{new_subject} | {r['relation']} | {new_object}"
        full.append({
            **base,
            "triple_text": entity_triple,
            "sentence": r["sentence"],
            "subject": new_subject, "relation": r["relation"], "object": new_object,
            "corruption_type": "entity",
            "corrupted_slot": corrupted_slot,
            "original_subject": r["subject"], "original_object": r["object"],
            "label": 0,
        })

        # relation_corruption: relation을 치환
        new_relation = other(relation_pool, r["relation"])
        relation_triple = f"{r['subject']} | {new_relation} | {r['object']}"
        full.append({
            **base,
            "triple_text": relation_triple,
            "sentence": r["sentence"],
            "subject": r["subject"], "relation": new_relation, "object": r["object"],
            "corruption_type": "relation",
            "original_relation": r["relation"],
            "label": 0,
        })

    rng.shuffle(full)
    return full
