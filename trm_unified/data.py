import json
import os
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Tuple


def iter_json_records(path: str) -> Iterable[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        first = f.read(1)
        while first and first.isspace():
            first = f.read(1)
        if not first:
            return
        f.seek(0)
        if first == '[':
            arr = json.load(f)
            for ex in arr:
                yield ex
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def load_kb_map(entities_txt: str) -> Dict[str, int]:
    kb2idx = {}
    with open(entities_txt, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            parts = line.rstrip('\n').split('\t')
            if parts:
                kb2idx[parts[0]] = i
    return kb2idx


def load_rel_map(relations_txt: str) -> Dict[str, int]:
    rel2idx = {}
    with open(relations_txt, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            rel = line.strip()
            if rel:
                rel2idx[rel] = i
    return rel2idx


def build_adj_from_tuples(tuples, rel2idx=None):
    adj = defaultdict(list)
    for s, r, o in tuples:
        try:
            s_id = int(s)
            o_id = int(o)
        except Exception:
            continue
        if isinstance(r, int):
            r_id = int(r)
        elif isinstance(r, str) and rel2idx is not None:
            if r not in rel2idx:
                continue
            r_id = rel2idx[r]
        else:
            continue
        adj[s_id].append((r_id, o_id))
    return adj


def mine_valid_paths(
    tuples,
    start_entities: List[int],
    goal_entities: List[int],
    max_steps: int = 4,
    max_paths: int = 4,
    max_neighbors: int = 128,
):
    if not tuples or not start_entities or not goal_entities:
        return []

    goal_set = set(int(x) for x in goal_entities)
    adj = build_adj_from_tuples(tuples)

    found = []
    for s in start_entities:
        q = deque()
        q.append((int(s), [int(s)], []))

        while q and len(found) < max_paths:
            cur, visited, triples = q.popleft()
            if len(triples) >= max_steps:
                continue
            cand = adj.get(cur, [])
            if max_neighbors > 0 and len(cand) > max_neighbors:
                cand = cand[:max_neighbors]
            for r, nxt in cand:
                if nxt in visited:
                    continue
                new_triples = triples + [[cur, int(r), int(nxt)]]
                if nxt in goal_set:
                    found.append(new_triples)
                    if len(found) >= max_paths:
                        break
                q.append((int(nxt), visited + [int(nxt)], new_triples))
    return found


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_line_offsets(jsonl_path: str, is_main: bool = True) -> List[int]:
    offsets = []
    with open(jsonl_path, "rb") as f:
        off = f.tell()
        line = f.readline()
        while line:
            offsets.append(off)
            off = f.tell()
            line = f.readline()
    if is_main:
        print(f"✅ Indexed {len(offsets)} lines from {jsonl_path}")
    return offsets


def read_jsonl_by_offset(jsonl_path: str, offsets: List[int], idx: int) -> dict:
    with open(jsonl_path, "rb") as f:
        f.seek(offsets[idx])
        line = f.readline()
    return json.loads(line.decode("utf-8"))


def _normalize_cwq(
    ex: dict,
    kb2idx: Dict[str, int],
    max_steps: int,
    max_paths: int,
    max_neighbors: int,
    mine_paths: bool = True,
) -> dict:
    entities = [int(x) for x in ex.get('entities', []) if isinstance(x, int)]
    answers = ex.get('answers', [])
    answers_cid = [int(x) for x in ex.get('answers_cid', []) if isinstance(x, int)]
    tuples = ex.get('subgraph', {}).get('tuples', [])
    valid_paths = ex.get('valid_paths', [])

    # Fallback for CWQ raws shaped like WebQSP: derive answers_cid/valid_paths.
    if (not answers_cid) and isinstance(answers, list):
        for a in answers:
            kb_id = a.get('kb_id') if isinstance(a, dict) else None
            if kb_id in kb2idx:
                answers_cid.append(int(kb2idx[kb_id]))
    if mine_paths and (not valid_paths) and tuples and entities and answers_cid:
        valid_paths = mine_valid_paths(
            tuples=tuples,
            start_entities=entities,
            goal_entities=answers_cid,
            max_steps=max_steps,
            max_paths=max_paths,
            max_neighbors=max_neighbors,
        )
    if not mine_paths:
        valid_paths = []

    relation_paths = [[int(step[1]) for step in p if isinstance(step, list) and len(step) == 3] for p in valid_paths]

    out = {
        'orig_id': ex.get('orig_id', ex.get('id', '')),
        'question': ex.get('question', ''),
        'entities': entities,
        'answers_cid': answers_cid,
        'answers': answers,
        'subgraph': {'tuples': tuples},
        'valid_paths': valid_paths,
        'relation_paths': relation_paths,
    }
    if max_steps > 0 and out['valid_paths']:
        out['valid_paths'] = [p[:max_steps] for p in out['valid_paths']]
        out['relation_paths'] = [
            [int(step[1]) for step in p if isinstance(step, list) and len(step) == 3]
            for p in out['valid_paths']
        ]
    return out


def _normalize_webqsp(
    ex: dict,
    kb2idx: Dict[str, int],
    max_steps: int,
    max_paths: int,
    max_neighbors: int,
    mine_paths: bool = True,
) -> dict:
    answers = ex.get('answers', [])
    answers_cid = []
    for a in answers:
        kb_id = a.get('kb_id') if isinstance(a, dict) else None
        if kb_id in kb2idx:
            answers_cid.append(int(kb2idx[kb_id]))

    entities = [int(x) for x in ex.get('entities', []) if isinstance(x, int)]
    tuples = ex.get('subgraph', {}).get('tuples', [])
    valid_paths = []
    if mine_paths:
        valid_paths = mine_valid_paths(
            tuples=tuples,
            start_entities=entities,
            goal_entities=answers_cid,
            max_steps=max_steps,
            max_paths=max_paths,
            max_neighbors=max_neighbors,
        )

    relation_paths = [[int(step[1]) for step in p if isinstance(step, list) and len(step) == 3] for p in valid_paths]

    return {
        'orig_id': ex.get('id', ''),
        'question': ex.get('question', ''),
        'entities': entities,
        'answers_cid': answers_cid,
        'answers': answers,
        'subgraph': {'tuples': tuples},
        'valid_paths': valid_paths,
        'relation_paths': relation_paths,
    }


def preprocess_split(
    dataset: str,
    input_path: str,
    output_path: str,
    entities_txt: str,
    max_steps: int = 4,
    max_paths: int = 4,
    max_neighbors: int = 128,
    mine_paths: bool = True,
    require_valid_paths: bool = True,
):
    dataset = dataset.lower()
    kb2idx = load_kb_map(entities_txt)

    kept = 0
    total = 0
    with open(output_path, 'w', encoding='utf-8') as w:
        for ex in iter_json_records(input_path):
            total += 1
            if dataset == 'cwq':
                obj = _normalize_cwq(ex, kb2idx, max_steps, max_paths, max_neighbors, mine_paths=mine_paths)
            elif dataset == 'webqsp':
                obj = _normalize_webqsp(ex, kb2idx, max_steps, max_paths, max_neighbors, mine_paths=mine_paths)
            else:
                raise ValueError(f'Unsupported dataset: {dataset}')

            if require_valid_paths and not obj['valid_paths']:
                continue
            w.write(json.dumps(obj, ensure_ascii=False) + '\n')
            kept += 1

    return {'total': total, 'kept': kept, 'output': output_path}
