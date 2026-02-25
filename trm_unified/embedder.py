import json
import os
import hashlib
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

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


def load_id_list(path: str) -> List[str]:
    ids = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            ids.append(line.split('\t')[0].strip())
    return ids


def load_entity_name_map(path: str) -> Dict[str, str]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def process_relation_name(rel_id: str) -> str:
    # GNN-RAG style: use the most specific relation token.
    name = rel_id.split('.')[-1] if '.' in rel_id else rel_id
    return name.replace('_', ' ')


def process_relation_name_gnn_exact(rel_id: str) -> str:
    # GNN-RAG gnn/dataset_load.py build_rel_words:
    # words = fields[-2].split('_') + fields[-1].split('_')
    rel = (rel_id or "").strip()
    if not rel:
        return "UNK"
    fields = rel.split(".")
    if len(fields) >= 2:
        words = fields[-2].split("_") + fields[-1].split("_")
        words = [w for w in words if w]
        return " ".join(words) if words else "UNK"
    return rel.replace("_", " ")


def collect_questions_and_ids(jsonl_path: str) -> Tuple[List[str], List[str]]:
    qs: List[str] = []
    qids: List[str] = []
    for ex in iter_json_records(jsonl_path):
        q = (
            ex.get('question')
            or ex.get('Question')
            or ex.get('query')
            or ex.get('question_text')
            or ''
        )
        qid = ex.get('orig_id', ex.get('id', ''))
        qs.append(str(q).strip())
        qids.append(str(qid))
    return qs, qids


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    s = (last_hidden_state * mask).sum(dim=1)
    d = mask.sum(dim=1).clamp(min=1e-6)
    return s / d


def _hash_vec(text: str, dim: int) -> np.ndarray:
    toks = (text or "").strip().split()
    if not toks:
        toks = ["<empty>"]
    v = np.zeros((dim,), dtype=np.float32)
    for t in toks:
        h = hashlib.sha1(t.encode("utf-8")).digest()
        for i in range(0, len(h), 2):
            slot = ((h[i] << 8) + h[i + 1]) % dim
            sign = 1.0 if (h[i] & 1) else -1.0
            v[slot] += sign
    n = np.linalg.norm(v) + 1e-12
    return v / n


def encode_texts_local_hash(texts: List[str], dim: int = 256, desc: str = "embed(local-hash)") -> np.ndarray:
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    return np.stack([_hash_vec(x, dim) for x in tqdm(texts, desc=desc, unit='txt')]).astype(np.float32)


def encode_texts(
    model_name: str,
    texts: List[str],
    batch_size: int,
    max_length: int,
    device: str,
    embed_gpus: str = "",
    prefix: str = "",
    backend: str = "auto",
    desc: str = "embed",
) -> np.ndarray:
    if prefix and (not prefix.endswith(" ")):
        prefix = prefix + " "
    prepared = [f"{prefix}{str(t)}" for t in texts]

    if (model_name or "").strip().lower() in {"local-hash", "local-simple", "local"}:
        return encode_texts_local_hash(prepared, desc=f"{desc}(local-hash)")

    selected_backend = (backend or "auto").strip().lower()
    if selected_backend in {"auto", "sentence_transformers", "sentence-transformers", "st"}:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            st_device = device
            gpu_ids = _parse_gpu_ids(embed_gpus)
            if torch.cuda.is_available() and gpu_ids:
                st_device = f"cuda:{gpu_ids[0]}"

            st_model = SentenceTransformer(model_name, device=st_device)
            emb = st_model.encode(
                prepared,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            return np.asarray(emb, dtype=np.float32)
        except Exception as e:
            # Fallback to transformers mean-pool path if sentence-transformers
            # is unavailable or fails to load this model.
            print(
                "⚠️ sentence-transformers backend unavailable; "
                f"falling back to transformers backend ({e})"
            )

    # Transformers mean-pool backend (legacy path)

    gpu_ids = _parse_gpu_ids(embed_gpus)
    run_device = device
    if torch.cuda.is_available() and gpu_ids:
        run_device = f"cuda:{gpu_ids[0]}"

    try:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        mdl = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(run_device)
        if torch.cuda.is_available() and len(gpu_ids) > 1:
            mdl = torch.nn.DataParallel(mdl, device_ids=gpu_ids, output_device=gpu_ids[0])
    except Exception as e:
        raise RuntimeError(
            f"failed to load embedding model '{model_name}' with backend='{selected_backend}' "
            f"(device='{run_device}'): {e}. "
            "Install/check model dependencies or set an explicit valid model name."
        ) from e
    mdl.eval()

    out = []
    total_batches = (len(texts) + batch_size - 1) // batch_size if batch_size > 0 else 0
    with torch.no_grad():
        for i in tqdm(range(0, len(prepared), batch_size), total=total_batches, desc=desc, unit='batch'):
            chunk = prepared[i:i + batch_size]
            t = tok(chunk, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
            t = {k: v.to(run_device, non_blocking=True) for k, v in t.items()}
            out_obj = mdl(**t, return_dict=False)
            h = out_obj[0] if isinstance(out_obj, (tuple, list)) else out_obj.last_hidden_state
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
    embed_gpus: str = "",
    entity_names_json: str = "",
    embed_style: str = "default",
    query_prefix: str = "",
    passage_prefix: str = "",
    embed_backend: str = "auto",
    save_lists: bool = True,
    test_jsonl: str = "",
):
    os.makedirs(out_dir, exist_ok=True)

    style = (embed_style or "default").strip().lower()
    gnn_style = style in {"gnn_rag", "gnn-rag", "paperstyle", "legacy"}
    gnn_exact_style = style in {
        "gnn_exact",
        "gnn-rag-gnn",
        "gnn_rag_gnn",
        "gnn_rag_gnn_exact",
        "gnn_gnn_exact",
    }

    if gnn_exact_style:
        ent_ids = load_id_list(entities_txt)
        rel_ids = load_id_list(relations_txt)
        # GNN-RAG gnn path does not use query:/passage: prefixes and builds
        # relation text from the last two dotted fields.
        # Keep entity surface text minimal (ID itself) to avoid extra supervision.
        ent_texts = list(ent_ids)
        rel_texts = [process_relation_name_gnn_exact(rid) for rid in rel_ids]
        if (embed_backend or "auto").strip().lower() == "auto":
            embed_backend = "sentence_transformers"
        # Force exact GNN-RAG gnn behavior: no query/passage prefixes.
        query_prefix = ""
        passage_prefix = ""
    elif gnn_style:
        ent_ids = load_id_list(entities_txt)
        rel_ids = load_id_list(relations_txt)
        name_map = load_entity_name_map(entity_names_json)
        ent_texts = [name_map.get(eid, eid) for eid in ent_ids]
        rel_texts = [process_relation_name(rid) for rid in rel_ids]
        # E5 conventions.
        if not query_prefix:
            query_prefix = "query: "
        if not passage_prefix:
            passage_prefix = "passage: "
        if (embed_backend or "auto").strip().lower() == "auto":
            # Match GNN-RAG style as closely as possible.
            embed_backend = "sentence_transformers"
    else:
        ent_texts = read_text_lines(entities_txt, mode='entity')
        rel_texts = read_text_lines(relations_txt, mode='relation')
        ent_ids = load_id_list(entities_txt)
        rel_ids = load_id_list(relations_txt)

    q_train, q_train_ids = collect_questions_and_ids(train_jsonl)
    q_dev, q_dev_ids = collect_questions_and_ids(dev_jsonl)

    ent = encode_texts(
        model_name, ent_texts, batch_size, max_length, device,
        embed_gpus=embed_gpus, prefix=passage_prefix, backend=embed_backend, desc='embed:entity'
    )
    rel = encode_texts(
        model_name, rel_texts, batch_size, max_length, device,
        embed_gpus=embed_gpus, prefix=passage_prefix, backend=embed_backend, desc='embed:relation'
    )
    qtr = encode_texts(
        model_name, q_train, batch_size, max_length, device,
        embed_gpus=embed_gpus, prefix=query_prefix, backend=embed_backend, desc='embed:query_train'
    )
    qdv = encode_texts(
        model_name, q_dev, batch_size, max_length, device,
        embed_gpus=embed_gpus, prefix=query_prefix, backend=embed_backend, desc='embed:query_dev'
    )

    np.save(os.path.join(out_dir, 'entity_embeddings.npy'), ent)
    np.save(os.path.join(out_dir, 'relation_embeddings.npy'), rel)
    np.save(os.path.join(out_dir, 'query_train.npy'), qtr)
    np.save(os.path.join(out_dir, 'query_dev.npy'), qdv)

    if test_jsonl and os.path.exists(test_jsonl):
        q_test, q_test_ids = collect_questions_and_ids(test_jsonl)
        qte = encode_texts(
            model_name, q_test, batch_size, max_length, device,
            embed_gpus=embed_gpus, prefix=query_prefix, backend=embed_backend, desc='embed:query_test'
        )
        np.save(os.path.join(out_dir, 'query_test.npy'), qte)
        if save_lists:
            _save_lines(os.path.join(out_dir, 'query_test.txt'), q_test)
            _save_lines(os.path.join(out_dir, 'query_test_ids.txt'), q_test_ids)

    meta = {
        'model_name': model_name,
        'embed_style': style,
        'embed_backend': embed_backend,
        'relation_text_mode': 'gnn_exact_last2' if gnn_exact_style else ('gnn_rag_last1' if gnn_style else 'default'),
        'query_prefix': query_prefix,
        'passage_prefix': passage_prefix,
        'entity_shape': list(ent.shape),
        'relation_shape': list(rel.shape),
        'query_train_shape': list(qtr.shape),
        'query_dev_shape': list(qdv.shape),
    }
    with open(os.path.join(out_dir, 'embedding_meta.json'), 'w', encoding='utf-8') as w:
        json.dump(meta, w, ensure_ascii=False, indent=2)
    if save_lists:
        _save_lines(os.path.join(out_dir, 'entity_ids.txt'), ent_ids)
        _save_lines(os.path.join(out_dir, 'relation_ids.txt'), rel_ids)
        _save_lines(os.path.join(out_dir, 'entity_names_used.txt'), ent_texts)
        _save_lines(os.path.join(out_dir, 'relation_names_used.txt'), rel_texts)
        _save_lines(os.path.join(out_dir, 'query_train.txt'), q_train)
        _save_lines(os.path.join(out_dir, 'query_dev.txt'), q_dev)
        _save_lines(os.path.join(out_dir, 'query_train_ids.txt'), q_train_ids)
        _save_lines(os.path.join(out_dir, 'query_dev_ids.txt'), q_dev_ids)

    return meta


def _parse_gpu_ids(embed_gpus: str) -> List[int]:
    raw = (embed_gpus or "").strip()
    if not raw:
        return []
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _save_lines(path: str, items: List[str]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for x in items:
            f.write(f"{x}\n")
