import json
import os
from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .data import iter_json_records


def read_text_lines(path: str, mode: str) -> List[str]:
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            parts = line.split('\t')
            if mode == 'entity':
                if len(parts) >= 2 and parts[1].strip():
                    texts.append(parts[1].strip())
                else:
                    texts.append(parts[0].strip())
            else:
                texts.append(parts[0].strip())
    return texts


def collect_questions(jsonl_path: str) -> List[str]:
    qs = []
    for ex in iter_json_records(jsonl_path):
        q = ex.get('question', '')
        qs.append(q if q else '')
    return qs


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    s = (last_hidden_state * mask).sum(dim=1)
    d = mask.sum(dim=1).clamp(min=1e-6)
    return s / d


def encode_texts(model_name: str, texts: List[str], batch_size: int, max_length: int, device: str) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    mdl.eval()

    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            t = tok(chunk, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(device)
            h = mdl(**t).last_hidden_state
            emb = mean_pool(h, t['attention_mask'])
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            out.append(emb.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0) if out else np.zeros((0, 1), dtype=np.float32)


def build_embeddings(
    model_name: str,
    entities_txt: str,
    relations_txt: str,
    train_jsonl: str,
    dev_jsonl: str,
    out_dir: str,
    batch_size: int = 64,
    max_length: int = 128,
    device: str = 'cuda',
):
    os.makedirs(out_dir, exist_ok=True)

    ent_texts = read_text_lines(entities_txt, mode='entity')
    rel_texts = read_text_lines(relations_txt, mode='relation')
    q_train = collect_questions(train_jsonl)
    q_dev = collect_questions(dev_jsonl)

    ent = encode_texts(model_name, ent_texts, batch_size, max_length, device)
    rel = encode_texts(model_name, rel_texts, batch_size, max_length, device)
    qtr = encode_texts(model_name, q_train, batch_size, max_length, device)
    qdv = encode_texts(model_name, q_dev, batch_size, max_length, device)

    np.save(os.path.join(out_dir, 'entity_embeddings.npy'), ent)
    np.save(os.path.join(out_dir, 'relation_embeddings.npy'), rel)
    np.save(os.path.join(out_dir, 'query_train.npy'), qtr)
    np.save(os.path.join(out_dir, 'query_dev.npy'), qdv)

    meta = {
        'model_name': model_name,
        'entity_shape': list(ent.shape),
        'relation_shape': list(rel.shape),
        'query_train_shape': list(qtr.shape),
        'query_dev_shape': list(qdv.shape),
    }
    with open(os.path.join(out_dir, 'embedding_meta.json'), 'w', encoding='utf-8') as w:
        json.dump(meta, w, ensure_ascii=False, indent=2)
    return meta
