#!/usr/bin/env python3
import argparse
import json
import sys
import urllib.parse
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trm_unified.data import iter_json_records, mine_valid_paths


QUESTION_FIELD_CANDIDATES = [
    "question",
    "query",
    "prompt",
    "question_text",
    "hop2_question_multi",
    "hop2_question",
    "hop1_question_multi",
    "hop1_question",
]
SEED_FIELD_CANDIDATES = [
    "entities",
    "seed_entities",
    "start_entities",
    "topic_entities",
    "question_entities",
]
ANSWER_FIELD_CANDIDATES = [
    "answers",
    "answer",
    "answer_entities",
    "target_entities",
    "targets",
]
TRIPLE_FIELD_CANDIDATES = [
    "subgraph.tuples",
    "subgraph.triples",
    "subgraph.edges",
    "tuples",
    "triples",
    "edges",
    "subgraph",
    "graph",
]
DICT_TRIPLE_KEY_CANDIDATES: List[Tuple[str, str, str]] = [
    ("s", "r", "o"),
    ("subject", "relation", "object"),
    ("head", "relation", "tail"),
    ("source", "relation", "target"),
    ("src", "rel", "dst"),
]
MEDHOP_ID_PAT = re.compile(r"\b(?:DB\d{5}|[A-NR-Z][0-9][A-Z0-9]{3}[0-9](?:-\d+)?)\b")


def _split_csv(raw: str) -> List[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _ensure_list(raw: Any) -> List[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, tuple):
        return list(raw)
    return [raw]


def _get_nested(ex: Dict[str, Any], dotted_key: str) -> Any:
    cur: Any = ex
    for part in dotted_key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _first_present(ex: Dict[str, Any], candidates: Sequence[str]) -> Any:
    for key in candidates:
        value = _get_nested(ex, key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, tuple, dict)) and not value:
            continue
        return value
    return None


def _stringify_entity(raw: Any) -> str:
    if isinstance(raw, dict):
        for key in ("kb_id", "entity_id", "id", "entity", "name", "text", "label"):
            value = raw.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return json.dumps(raw, ensure_ascii=False, sort_keys=True)
    return str(raw).strip()


def _decode_text(raw: Any) -> str:
    text = _stringify_entity(raw)
    text = urllib.parse.unquote(text)
    text = text.strip().strip(">").strip()
    while text.startswith("[") and text.endswith("]") and len(text) >= 2:
        text = text[1:-1].strip()
    return text


def _normalize_answers(raw_answers: Any, target_type: str = "") -> Tuple[List[str], List[Dict[str, str]]]:
    entity_ids: List[str] = []
    answer_meta: List[Dict[str, str]] = []
    for item in _ensure_list(raw_answers):
        text = _stringify_entity(item)
        if not text:
            continue
        kb_id = f"{target_type}::{text}" if target_type else text
        entity_ids.append(kb_id)
        answer_meta.append({"kb_id": kb_id, "text": text})
    return entity_ids, answer_meta


def _normalize_seed_entities(raw_entities: Any, entity_type: str = "") -> List[str]:
    out: List[str] = []
    for item in _ensure_list(raw_entities):
        text = _stringify_entity(item)
        if not text:
            continue
        out.append(f"{entity_type}::{text}" if entity_type else text)
    return out


def _unwrap_triple_container(raw: Any) -> List[Any]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        for key in ("tuples", "triples", "edges"):
            value = raw.get(key)
            if isinstance(value, list):
                return value
        return []
    if isinstance(raw, list):
        return raw
    return []


def _extract_triples_generic(
    ex: Dict[str, Any],
    triple_field: str,
    triple_keys: Tuple[str, str, str],
) -> List[Tuple[str, str, str]]:
    raw = _get_nested(ex, triple_field) if triple_field else _first_present(ex, TRIPLE_FIELD_CANDIDATES)
    items = _unwrap_triple_container(raw)
    triples: List[Tuple[str, str, str]] = []
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            s, r, o = item[0], item[1], item[2]
        elif isinstance(item, dict):
            found = None
            candidate_keys = [triple_keys] + [keys for keys in DICT_TRIPLE_KEY_CANDIDATES if keys != triple_keys]
            for s_key, r_key, o_key in candidate_keys:
                if s_key in item and r_key in item and o_key in item:
                    found = (item[s_key], item[r_key], item[o_key])
                    break
            if found is None:
                continue
            s, r, o = found
        else:
            continue
        s_txt = _stringify_entity(s)
        r_txt = str(r).strip()
        o_txt = _stringify_entity(o)
        if s_txt and r_txt and o_txt:
            triples.append((s_txt, r_txt, o_txt))
    return triples


def _generic_question(ex: Dict[str, Any], explicit_field: str) -> str:
    raw = _get_nested(ex, explicit_field) if explicit_field else _first_present(ex, QUESTION_FIELD_CANDIDATES)
    return str(raw or "").strip()


def _generic_id(ex: Dict[str, Any], index: int, explicit_field: str) -> str:
    if explicit_field:
        raw = _get_nested(ex, explicit_field)
    else:
        raw = _first_present(ex, ["orig_id", "id", "qid", "question_id"])
    return str(raw if raw is not None else index)


def _convert_generic_record(ex: Dict[str, Any], index: int, args: argparse.Namespace) -> Dict[str, Any]:
    question = _generic_question(ex, args.question_field)
    orig_id = _generic_id(ex, index, args.id_field)
    seed_raw = _get_nested(ex, args.seed_field) if args.seed_field else _first_present(ex, SEED_FIELD_CANDIDATES)
    answer_raw = _get_nested(ex, args.answer_field) if args.answer_field else _first_present(ex, ANSWER_FIELD_CANDIDATES)
    seeds = _normalize_seed_entities(seed_raw)
    answer_ids, answers = _normalize_answers(answer_raw)
    triples = _extract_triples_generic(ex, args.triple_field, (args.triple_subject_key, args.triple_relation_key, args.triple_object_key))
    if not question:
        raise ValueError(f"record {orig_id} is missing a question field")
    if not seeds:
        raise ValueError(f"record {orig_id} is missing seed entities")
    if not answer_ids:
        raise ValueError(f"record {orig_id} is missing answers")
    if not triples:
        raise ValueError(f"record {orig_id} is missing subgraph triples")
    return {
        "orig_id": orig_id,
        "question": question,
        "seed_entities": seeds,
        "answer_entities": answer_ids,
        "answers": answers,
        "triples": triples,
    }


def _pick_biohopr_question(ex: Dict[str, Any]) -> str:
    has_hop2 = bool(str(ex.get("hop2", "")).strip())
    if has_hop2:
        for key in ("hop2_question_multi", "hop2_question", "question"):
            value = str(ex.get(key, "")).strip()
            if value:
                return value
    for key in ("hop1_question_multi", "hop1_question", "question"):
        value = str(ex.get(key, "")).strip()
        if value:
            return value
    return ""


def _primekg_entity_node(text: str) -> str:
    return f"primekg_ent::{text}"


def _primekg_relation_node(text: str) -> str:
    return f"primekg_relans::{text}"


def _normalize_primekg_answers(answer_raw: Any, triple_rows: List[Tuple[str, str, str]]) -> Tuple[List[str], List[Dict[str, str]]]:
    relation_texts = {r for _, r, _ in triple_rows}
    entity_texts = {s for s, _, _ in triple_rows} | {o for _, _, o in triple_rows}
    answer_ids: List[str] = []
    answer_meta: List[Dict[str, str]] = []
    for item in _ensure_list(answer_raw):
        text = _decode_text(item)
        if not text:
            continue
        if text in relation_texts:
            kb_id = _primekg_relation_node(text)
        elif text in entity_texts:
            kb_id = _primekg_entity_node(text)
        else:
            kb_id = f"primekg_ans::{text}"
        answer_ids.append(kb_id)
        answer_meta.append({"kb_id": kb_id, "text": text})
    return answer_ids, answer_meta


def _primekg_question_from_record(ex: Dict[str, Any], triple_rows: List[Tuple[str, str, str]]) -> str:
    for key in ("generated_question", "generated_text", "question"):
        value = str(ex.get(key, "")).strip()
        if value:
            return value

    for key in ("orig_output", "prompt"):
        value = " ".join(str(ex.get(key, "")).split())
        if not value:
            continue
        match = re.search(r"question is:\s*(.+?\?)", value, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        q_matches = re.findall(r"([A-Z][^?]{8,}\?)", value)
        if q_matches:
            return q_matches[-1].strip()

    facts = "; ".join(f"{s} {r} {o}" for s, r, o in triple_rows[:4])
    return f"Given the PrimeKG facts {facts}, what is the best answer?"


def _convert_primekgqa_record(ex: Dict[str, Any], index: int) -> Dict[str, Any]:
    orig_id = str(ex.get("id", index))
    raw_rows = _ensure_list(ex.get("value"))
    triple_rows: List[Tuple[str, str, str]] = []
    for row in raw_rows:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        s_txt = _decode_text(row[0])
        r_txt = _decode_text(row[1])
        o_txt = _decode_text(row[2])
        if s_txt and r_txt and o_txt:
            triple_rows.append((s_txt, r_txt, o_txt))
    question = _primekg_question_from_record(ex, triple_rows)

    if not question:
        raise ValueError(f"PrimeKGQA record {orig_id} is missing a question field")
    if not triple_rows:
        raise ValueError(f"PrimeKGQA record {orig_id} is missing triples")

    answer_ids, answers = _normalize_primekg_answers(ex.get("answer", []), triple_rows)
    if not answer_ids:
        raise ValueError(f"PrimeKGQA record {orig_id} is missing answers")

    graph_triples: List[Tuple[str, str, str]] = []
    seed_entities: List[str] = []
    for s_txt, r_txt, o_txt in triple_rows:
        s_node = _primekg_entity_node(s_txt)
        o_node = _primekg_entity_node(o_txt)
        r_node = _primekg_relation_node(r_txt)
        seed_entities.extend([s_node, o_node])
        graph_triples.append((s_node, f"primekg_rel::{r_txt}", o_node))
        # Relation-as-answer bridge so the node-based reader can score relation answers too.
        graph_triples.append((s_node, "primekg_meta::has_relation", r_node))
        graph_triples.append((r_node, "primekg_meta::relation_target", o_node))

    unique_seeds = []
    seen_seeds = set(answer_ids)
    for ent in seed_entities:
        if ent in seen_seeds:
            continue
        if ent in unique_seeds:
            continue
        unique_seeds.append(ent)
    if not unique_seeds:
        unique_seeds = list(dict.fromkeys(seed_entities))

    return {
        "orig_id": orig_id,
        "question": question,
        "seed_entities": unique_seeds,
        "answer_entities": answer_ids,
        "answers": answers,
        "triples": graph_triples,
    }


def _extract_medhop_seed_and_relation(question: str) -> Tuple[str, str]:
    q = str(question or "").strip()
    ids = MEDHOP_ID_PAT.findall(q)
    seed = ids[-1] if ids else ""
    relation = q
    if seed:
        relation = relation.replace(seed, " ").strip()
    relation = relation.rstrip("?").strip().replace(" ", "_")
    if not relation:
        relation = "related_to"
    return seed, relation


def _extract_medhop_context_ids(context: str, *, limit: int = 128) -> List[str]:
    seen = set()
    out: List[str] = []
    for match in MEDHOP_ID_PAT.findall(str(context or "")):
        if match in seen:
            continue
        seen.add(match)
        out.append(match)
        if len(out) >= limit:
            break
    return out


def _convert_medhopqa_record(ex: Dict[str, Any], index: int) -> Dict[str, Any]:
    orig_id = str(ex.get("id", index))
    question = str(ex.get("question", "")).strip()
    if not question:
        raise ValueError(f"MedHopQA record {orig_id} is missing question")

    seed_text, relation = _extract_medhop_seed_and_relation(question)
    if not seed_text:
        raise ValueError(f"MedHopQA record {orig_id} is missing a seed entity in the question")

    seed_node = f"medhop_ent::{seed_text}"
    choices = [_decode_text(item) for item in _ensure_list(ex.get("choices", [])) if _decode_text(item)]
    answer_ids, answers = _normalize_answers(ex.get("answer", []), target_type="medhop_ent")
    if not choices:
        raise ValueError(f"MedHopQA record {orig_id} is missing answer choices")
    if not answer_ids:
        raise ValueError(f"MedHopQA record {orig_id} is missing answers")

    graph_triples: List[Tuple[str, str, str]] = []
    choice_rel = f"medhop_rel::{relation}"
    for choice in choices:
        graph_triples.append((seed_node, choice_rel, f"medhop_ent::{choice}"))

    for ctx_id in _extract_medhop_context_ids(ex.get("context", "")):
        ctx_node = f"medhop_ent::{ctx_id}"
        graph_triples.append((seed_node, "medhop_rel::context_mentions", ctx_node))

    return {
        "orig_id": orig_id,
        "question": question,
        "seed_entities": [seed_node],
        "answer_entities": answer_ids,
        "answers": answers,
        "triples": graph_triples,
    }


def _convert_biohopr_record(ex: Dict[str, Any], index: int) -> Dict[str, Any]:
    orig_id = str(ex.get("id", index))
    question = _pick_biohopr_question(ex)
    hop1 = str(ex.get("hop1", "")).strip()
    hop1_type = str(ex.get("hop1_type", "entity")).strip() or "entity"
    hop2 = str(ex.get("hop2", "")).strip()
    hop2_type = str(ex.get("hop2_type", "entity")).strip() or "entity"
    target_type = str(ex.get("target_type", "entity")).strip() or "entity"
    rel_hop1 = str(ex.get("relation_hop1", "")).strip() or f"{hop1_type}_to_{target_type}"
    rel_hop2 = str(ex.get("relation_hop2", "")).strip() or f"{hop2_type}_to_{hop1_type}"

    answer_ids, answers = _normalize_answers(ex.get("answer", ex.get("answers", [])), target_type=target_type)
    if not question:
        raise ValueError(f"BioHopR record {orig_id} is missing a question field")
    if not hop1:
        raise ValueError(f"BioHopR record {orig_id} is missing hop1")
    if not answer_ids:
        raise ValueError(f"BioHopR record {orig_id} is missing answers")

    hop1_entity = f"{hop1_type}::{hop1}"
    triples: List[Tuple[str, str, str]] = []
    if hop2:
        hop2_entity = f"{hop2_type}::{hop2}"
        triples.append((hop2_entity, f"biohopr::{rel_hop2}", hop1_entity))
        for ans in answer_ids:
            triples.append((hop1_entity, f"biohopr::{rel_hop1}", ans))
        seed_entities = [hop2_entity]
    else:
        for ans in answer_ids:
            triples.append((hop1_entity, f"biohopr::{rel_hop1}", ans))
        seed_entities = [hop1_entity]

    return {
        "orig_id": orig_id,
        "question": question,
        "seed_entities": seed_entities,
        "answer_entities": answer_ids,
        "answers": answers,
        "triples": triples,
    }


def _load_split(path: str) -> List[Dict[str, Any]]:
    if not path:
        return []
    return list(iter_json_records(path))


def _build_vocab(records_by_split: Dict[str, List[Dict[str, Any]]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    entity2id: Dict[str, int] = {}
    rel2id: Dict[str, int] = {}
    for records in records_by_split.values():
        for rec in records:
            for ent in rec["seed_entities"]:
                entity2id.setdefault(ent, len(entity2id))
            for ent in rec["answer_entities"]:
                entity2id.setdefault(ent, len(entity2id))
            for subj, rel, obj in rec["triples"]:
                entity2id.setdefault(subj, len(entity2id))
                entity2id.setdefault(obj, len(entity2id))
                rel2id.setdefault(rel, len(rel2id))
    return entity2id, rel2id


def _relation_paths_from_valid_paths(valid_paths: List[List[List[int]]]) -> List[List[int]]:
    out: List[List[int]] = []
    for path in valid_paths:
        out.append([int(step[1]) for step in path if isinstance(step, list) and len(step) >= 3])
    return out


def _normalize_processed_record(
    rec: Dict[str, Any],
    entity2id: Dict[str, int],
    rel2id: Dict[str, int],
    *,
    max_steps: int,
    max_paths: int,
    max_neighbors: int,
) -> Dict[str, Any]:
    int_triples = [[entity2id[s], rel2id[r], entity2id[o]] for s, r, o in rec["triples"]]
    int_seeds = [entity2id[x] for x in rec["seed_entities"] if x in entity2id]
    int_answers = [entity2id[x] for x in rec["answer_entities"] if x in entity2id]
    valid_paths = mine_valid_paths(
        tuples=int_triples,
        start_entities=int_seeds,
        goal_entities=int_answers,
        max_steps=max_steps,
        max_paths=max_paths,
        max_neighbors=max_neighbors,
    )
    return {
        "orig_id": rec["orig_id"],
        "question": rec["question"],
        "entities": int_seeds,
        "answers_cid": int_answers,
        "answers": rec["answers"],
        "subgraph": {"tuples": int_triples},
        "valid_paths": valid_paths,
        "relation_paths": _relation_paths_from_valid_paths(valid_paths),
    }


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_vocab(path: Path, items: Dict[str, int], *, entity_mode: bool) -> None:
    ordered = sorted(items.items(), key=lambda kv: kv[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for key, idx in ordered:
            if entity_mode:
                text = key.split("::", 1)[1] if "::" in key else key
                handle.write(f"{key}\t{text}\n")
            else:
                handle.write(f"{key}\n")


def _convert_records(raw_records: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for idx, ex in enumerate(raw_records):
        if args.dataset == "biohopr":
            converted.append(_convert_biohopr_record(ex, idx))
        elif args.dataset == "primekgqa":
            converted.append(_convert_primekgqa_record(ex, idx))
        elif args.dataset == "medhopqa":
            converted.append(_convert_medhopqa_record(ex, idx))
        else:
            converted.append(_convert_generic_record(ex, idx, args))
    return converted


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert medical/custom KGQA data into repo-ready processed jsonl files")
    ap.add_argument("--dataset", choices=["generic", "primekgqa", "biohopr", "medhopqa"], required=True)
    ap.add_argument("--dataset-name", default="", help="Output dataset folder name; defaults to --dataset")
    ap.add_argument("--train-in", required=True)
    ap.add_argument("--dev-in", default="")
    ap.add_argument("--test-in", default="")
    ap.add_argument("--out-root", default=".")
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--max-paths", type=int, default=4)
    ap.add_argument("--max-neighbors", type=int, default=128)

    ap.add_argument("--question-field", default="")
    ap.add_argument("--id-field", default="")
    ap.add_argument("--seed-field", default="")
    ap.add_argument("--answer-field", default="")
    ap.add_argument("--triple-field", default="")
    ap.add_argument("--triple-subject-key", default="s")
    ap.add_argument("--triple-relation-key", default="r")
    ap.add_argument("--triple-object-key", default="o")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_name = (args.dataset_name or args.dataset).strip().lower()
    out_root = Path(args.out_root).resolve()
    dataset_root = out_root / "data" / dataset_name

    raw_by_split = {
        "train": _load_split(args.train_in),
        "dev": _load_split(args.dev_in),
        "test": _load_split(args.test_in),
    }
    converted_by_split = {
        split: _convert_records(records, args)
        for split, records in raw_by_split.items()
        if records
    }
    if "train" not in converted_by_split:
        raise ValueError("train split is required")

    entity2id, rel2id = _build_vocab(converted_by_split)
    processed_by_split = {
        split: [
            _normalize_processed_record(
                rec,
                entity2id,
                rel2id,
                max_steps=args.max_steps,
                max_paths=args.max_paths,
                max_neighbors=args.max_neighbors,
            )
            for rec in records
        ]
        for split, records in converted_by_split.items()
    }

    for split, records in processed_by_split.items():
        _write_jsonl(dataset_root / f"{split}.jsonl", records)

    _write_vocab(dataset_root / "entities.txt", entity2id, entity_mode=True)
    _write_vocab(dataset_root / "relations.txt", rel2id, entity_mode=False)

    meta = {
        "dataset": dataset_name,
        "splits": {split: len(records) for split, records in processed_by_split.items()},
        "num_entities": len(entity2id),
        "num_relations": len(rel2id),
        "output_dir": str(dataset_root),
    }
    (dataset_root / "conversion_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
