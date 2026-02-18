import importlib
import math
import os
import re
import json
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .data import build_adj_from_tuples, iter_json_records, load_kb_map, load_rel_map, read_jsonl_by_offset, build_line_offsets
from .tokenization import load_tokenizer


def _setup_wandb(args, is_main: bool):
    if not is_main:
        return None
    mode = str(getattr(args, "wandb_mode", "disabled") or "disabled").lower()
    if mode in {"off", "false", "none", "disabled", "disable"}:
        return None
    try:
        import wandb  # type: ignore
    except Exception as e:
        print(f"[warn] wandb import failed: {e}; continue without wandb logging")
        return None

    run = wandb.init(
        project=getattr(args, "wandb_project", None) or "graph-traverse",
        entity=getattr(args, "wandb_entity", None) or None,
        name=getattr(args, "wandb_run_name", None) or None,
        mode=mode,
        config={
            "dataset": os.path.basename(getattr(args, "train_json", "")),
            "model_impl": getattr(args, "model_impl", ""),
            "batch_size": int(getattr(args, "batch_size", 0)),
            "epochs": int(getattr(args, "epochs", 0)),
            "lr": float(getattr(args, "lr", 0.0)),
            "max_steps": int(getattr(args, "max_steps", 0)),
            "beam": int(getattr(args, "beam", 0)),
            "start_topk": int(getattr(args, "start_topk", 0)),
        },
    )
    return run


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def prune_rel_mix_cpu(q_emb, cand_rels, rel_mem, keep_k, rand_k, rand_pool_mult=4):
    if keep_k <= 0 and rand_k <= 0:
        return cand_rels
    if len(cand_rels) <= keep_k + rand_k:
        return cand_rels
    r_ids = np.array(cand_rels, dtype=np.int64)
    r = l2_normalize_np(np.asarray(rel_mem[r_ids], dtype=np.float32))
    q = q_emb / (np.linalg.norm(q_emb) + 1e-12)
    score = r @ q
    idx_sorted = np.argsort(-score)
    keep = idx_sorted[:max(0, keep_k)].tolist()
    start = len(keep)
    end = min(len(idx_sorted), start + rand_pool_mult * max(0, rand_k))
    mid = idx_sorted[start:end]
    rand = []
    if rand_k > 0 and len(mid) > 0:
        rand = np.random.choice(mid, size=min(rand_k, len(mid)), replace=False).tolist()
    sel = list(dict.fromkeys(keep + rand))
    return [cand_rels[i] for i in sel]


class PathDataset(Dataset):
    def __init__(self, jsonl_path: str, max_steps: int):
        self.jsonl_path = jsonl_path
        self.max_steps = max_steps
        self.offsets = build_line_offsets(jsonl_path, is_main=True)
        self.flat = []
        for i in tqdm(range(len(self.offsets)), desc='Flattening'):
            ex = read_jsonl_by_offset(self.jsonl_path, self.offsets, i)
            vps = ex.get('valid_paths', [])
            for pidx in range(len(vps)):
                self.flat.append((i, pidx))

    def __len__(self):
        return len(self.flat)

    def __getitem__(self, idx):
        li, pi = self.flat[idx]
        ex = read_jsonl_by_offset(self.jsonl_path, self.offsets, li)
        path = ex.get('valid_paths', [])[pi][:self.max_steps]
        if not path:
            segs = [0, 0, 0]
        else:
            segs = [path[0][0]]
            for s, r, o in path:
                segs.append(r)
                segs.append(o)
        answers_cid = []
        for x in ex.get('answers_cid', []):
            try:
                answers_cid.append(int(x))
            except Exception:
                pass
        return {
            'q_text': ex.get('question', ''),
            'tuples': ex.get('subgraph', {}).get('tuples', []),
            'path_segments': segs,
            'answers_cid': answers_cid,
            'ex_line': li,
        }


def make_collate(
    tokenizer_name,
    rel_npy,
    q_npy,
    max_neighbors,
    prune_keep,
    prune_rand,
    max_q_len,
    max_steps,
    endpoint_aux=False,
    entity_vocab_size: Optional[int] = None,
):
    tok = load_tokenizer(tokenizer_name)
    rel_mem = np.load(rel_npy, mmap_mode='r')
    q_mem = np.load(q_npy, mmap_mode='r') if q_npy and os.path.exists(q_npy) else None

    def _can_reach_goal(adj, node: int, goals: set, rem_steps: int, memo: dict) -> bool:
        key = (int(node), int(rem_steps))
        if key in memo:
            return memo[key]
        n = int(node)
        if n in goals:
            memo[key] = True
            return True
        if rem_steps <= 0:
            memo[key] = False
            return False
        for _, nxt in adj.get(n, []):
            if _can_reach_goal(adj, int(nxt), goals, rem_steps - 1, memo):
                memo[key] = True
                return True
        memo[key] = False
        return False

    def _fn(batch):
        toks = tok([b['q_text'] for b in batch], padding=True, truncation=True, max_length=max_q_len, return_tensors='pt')
        sample_ctx = []
        for b in batch:
            adj = build_adj_from_tuples(b['tuples'])
            if q_mem is None:
                q_emb = np.zeros((rel_mem.shape[1],), dtype=np.float32)
            else:
                qi = b['ex_line'] if b['ex_line'] < q_mem.shape[0] else 0
                q_emb = np.asarray(q_mem[qi], dtype=np.float32)
            gold_set = set(int(x) for x in b.get('answers_cid', []))
            sample_ctx.append((adj, q_emb, gold_set, {}))

        seq_batches = []
        for t in range(max_steps):
            puzs, rels, labels, cmask, vmask, endpoint_targets = [], [], [], [], [], []
            for i, b in enumerate(batch):
                segs = b['path_segments']
                idx = t * 2
                if idx + 2 >= len(segs):
                    puzs.append([0]); rels.append([0]); labels.append(-100); cmask.append([False]); vmask.append(False); endpoint_targets.append([0.0])
                    continue
                cur = int(segs[idx])
                gold_r = int(segs[idx + 1])
                adj, q_emb, gold_set, reach_memo = sample_ctx[i]
                edges = adj.get(cur, [])
                if max_neighbors > 0 and len(edges) > max_neighbors:
                    edges = edges[:max_neighbors]
                rel_cands = []
                seen = set()
                for r, _ in edges:
                    r = int(r)
                    if r not in seen:
                        seen.add(r)
                        rel_cands.append(r)
                rel_cands = prune_rel_mix_cpu(q_emb, rel_cands, rel_mem, prune_keep, prune_rand)
                if gold_r not in rel_cands:
                    rel_cands.append(gold_r)
                np.random.shuffle(rel_cands)
                cur_pid = int(cur)
                if cur_pid < 0 or (entity_vocab_size is not None and cur_pid >= int(entity_vocab_size)):
                    cur_pid = 0
                puzs.append([cur_pid] * len(rel_cands))
                rels.append(rel_cands)
                labels.append(rel_cands.index(gold_r))
                cmask.append([True] * len(rel_cands))
                vmask.append(True)
                if endpoint_aux:
                    rem_steps = max(0, max_steps - (t + 1))
                    pos_rels = set()
                    if gold_set:
                        for r, nxt in edges:
                            rr = int(r)
                            nn = int(nxt)
                            if rr in pos_rels:
                                continue
                            if _can_reach_goal(adj, nn, gold_set, rem_steps, reach_memo):
                                pos_rels.add(rr)
                    endpoint_targets.append([1.0 if rr in pos_rels else 0.0 for rr in rel_cands])
                else:
                    endpoint_targets.append([0.0] * len(rel_cands))

            cmax = max(len(x) for x in puzs)
            B = len(batch)
            pt = torch.zeros((B, cmax), dtype=torch.long)
            rt = torch.zeros((B, cmax), dtype=torch.long)
            mt = torch.zeros((B, cmax), dtype=torch.bool)
            et = torch.zeros((B, cmax), dtype=torch.float32)
            for k in range(B):
                L = len(puzs[k])
                pt[k, :L] = torch.tensor(puzs[k], dtype=torch.long)
                rt[k, :L] = torch.tensor(rels[k], dtype=torch.long)
                mt[k, :L] = torch.tensor(cmask[k], dtype=torch.bool)
                et[k, :L] = torch.tensor(endpoint_targets[k], dtype=torch.float32)
            seq_batches.append({
                'puzzle_identifiers': pt,
                'relation_identifiers': rt,
                'candidate_mask': mt,
                'endpoint_targets': et,
                'labels': torch.tensor(labels, dtype=torch.long),
                'valid_mask': torch.tensor(vmask, dtype=torch.bool),
            })

        # Halt supervision:
        # - valid steps before the last hop: 0
        # - last valid hop of each path: 1
        # - padded steps: excluded from halt loss
        for t in range(max_steps):
            vm = seq_batches[t]['valid_mask']
            if t + 1 < max_steps:
                next_vm = seq_batches[t + 1]['valid_mask']
            else:
                next_vm = torch.zeros_like(vm)
            halt_targets = (vm & (~next_vm)).to(torch.float32)
            halt_mask = vm.clone()
            seq_batches[t]['halt_targets'] = halt_targets
            seq_batches[t]['halt_mask'] = halt_mask

        return {'input_ids': toks['input_ids'], 'attention_mask': toks['attention_mask'], 'seq_batches': seq_batches}

    return _fn


def _setup_ddp():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cpu')
        return True, rank, local_rank, world_size, device
    return False, 0, 0, 1, torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model(model_impl: str, trm_root: str, cfg: dict):
    if not os.path.isdir(trm_root):
        raise FileNotFoundError(
            f"TRM root not found: {trm_root}. "
            "Set --trm_root or TRM_ROOT to your TinyRecursiveModels path."
        )
    if trm_root not in os.sys.path:
        os.sys.path.append(trm_root)
    mod = importlib.import_module(f'models.recursive_reasoning.{model_impl}')
    cls = getattr(mod, 'TinyRecursiveReasoningModel_ACTV1')
    carry_cls = getattr(mod, 'TinyRecursiveReasoningModel_ACTV1Carry')
    return cls(cfg), carry_cls


def _clone_carry(carry):
    inner = carry.inner_carry
    inner_type = type(inner)
    carry_type = type(carry)
    inner_kwargs = {}
    for k in inner.__dataclass_fields__.keys():
        v = getattr(inner, k)
        inner_kwargs[k] = v.clone()
    current_data = {}
    for k, v in carry.current_data.items():
        current_data[k] = v.clone() if torch.is_tensor(v) else v
    return carry_type(
        inner_carry=inner_type(**inner_kwargs),
        steps=carry.steps.clone(),
        halted=carry.halted.clone(),
        current_data=current_data,
    )


def _parse_eval_example(ex: dict, kb2idx: Dict[str, int], rel2idx: Dict[str, int]):
    raw_tuples = ex.get('subgraph', {}).get('tuples', [])
    if not raw_tuples and 'subgraph_edges' in ex:
        raw_tuples = ex.get('subgraph_edges', [])

    final_tuples = []
    for s, r, o in raw_tuples:
        try:
            s_id = int(s)
            o_id = int(o)
        except Exception:
            continue
        if isinstance(r, int):
            r_id = int(r)
        elif isinstance(r, str):
            if r not in rel2idx:
                continue
            r_id = int(rel2idx[r])
        else:
            continue
        final_tuples.append((s_id, r_id, o_id))

    starts = []
    # Prefer canonical CID starts when available.
    for x in ex.get('entities_cid', []):
        try:
            starts.append(int(x))
        except Exception:
            pass

    if 'seed_entities' in ex:
        for se in ex.get('seed_entities', []):
            if isinstance(se, dict) and 'node_id' in se:
                try:
                    starts.append(int(se['node_id']))
                except Exception:
                    pass
    if not starts:
        for x in ex.get('entities', []):
            try:
                starts.append(int(x))
            except Exception:
                pass

    gold = set()
    for x in ex.get('answers_cid', []):
        try:
            gold.add(int(x))
        except Exception:
            pass
    for a in ex.get('answers', []):
        if isinstance(a, dict):
            kb_id = a.get('kb_id')
            if kb_id in kb2idx:
                gold.add(int(kb2idx[kb_id]))
    for a in ex.get('gold_answers', []):
        if isinstance(a, dict) and 'node_id' in a:
            try:
                gold.add(int(a['node_id']))
            except Exception:
                pass

    return final_tuples, starts, gold


def _load_entity_name_map(entity_names_json: str) -> Dict[str, str]:
    if not entity_names_json or not os.path.exists(entity_names_json):
        return {}
    try:
        with open(entity_names_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _load_entity_labels(entities_txt: str, entity_names_json: str = "") -> List[str]:
    name_map = _load_entity_name_map(entity_names_json)
    out = []
    with open(entities_txt, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            eid = parts[0].strip() if parts else ""
            name = parts[1].strip() if len(parts) > 1 else ""
            if eid in name_map and str(name_map[eid]).strip():
                name = str(name_map[eid]).strip()
            if name and name != eid:
                out.append(f"{eid}:{name}")
            else:
                out.append(eid)
    return out


def _load_relation_labels(relations_txt: str) -> List[str]:
    out = []
    with open(relations_txt, "r", encoding="utf-8") as f:
        for line in f:
            out.append(line.strip())
    return out


def _idx_to_label(labels: List[str], idx: int) -> str:
    i = int(idx)
    if 0 <= i < len(labels):
        return labels[i]
    return str(i)


def _find_gold_path(adj, starts: List[int], gold_set: set, max_steps: int):
    starts = [int(x) for x in starts]
    if not starts or not gold_set:
        return None

    for s in starts:
        if s in gold_set:
            return {"nodes": [s], "rels": []}

    q = deque()
    parent = {}  # node -> (prev_node, rel_id)
    dist = {}
    for s in starts:
        q.append(int(s))
        dist[int(s)] = 0

    target = None
    while q:
        cur = q.popleft()
        d = dist[cur]
        if d >= max_steps:
            continue
        for r, nxt in adj.get(cur, []):
            nxt = int(nxt)
            if nxt in dist:
                continue
            dist[nxt] = d + 1
            parent[nxt] = (cur, int(r))
            if nxt in gold_set:
                target = nxt
                q.clear()
                break
            q.append(nxt)

    if target is None:
        return None

    rels = []
    nodes = [int(target)]
    cur = int(target)
    while cur in parent:
        prev, rel = parent[cur]
        rels.append(int(rel))
        nodes.append(int(prev))
        cur = int(prev)
    nodes.reverse()
    rels.reverse()
    return {"nodes": nodes, "rels": rels}


def diagnose_oracle_reachability(
    eval_json: str,
    kb2idx: Dict[str, int],
    rel2idx: Dict[str, int],
    max_steps: int,
    max_neighbors: int,
    start_topk: int,
    eval_limit: int = -1,
):
    data = list(iter_json_records(eval_json))
    total = len(data) if eval_limit < 0 else min(len(data), eval_limit)
    steps_cap = max(0, int(max_steps))
    topk = max(1, int(start_topk))
    neighbor_cap = int(max_neighbors)

    usable = 0
    answer_in_subgraph = 0
    reachable_at_k = 0
    reachable_any = 0
    need_more_steps = 0
    no_start = 0
    no_gold = 0
    no_tuple = 0
    shortest_hops = []
    hop_hist = [0 for _ in range(steps_cap + 1)]

    for ex_idx in tqdm(range(total), desc='OracleDiag'):
        ex = data[ex_idx]
        tuples, starts_raw, gold = _parse_eval_example(ex, kb2idx, rel2idx)
        if not tuples:
            no_tuple += 1
            continue
        starts = list(dict.fromkeys(starts_raw))[:topk]
        if not starts:
            no_start += 1
            continue
        if not gold:
            no_gold += 1
            continue

        usable += 1
        nodes_in_subgraph = set()
        for s, _, o in tuples:
            nodes_in_subgraph.add(int(s))
            nodes_in_subgraph.add(int(o))
        if bool(gold & nodes_in_subgraph):
            answer_in_subgraph += 1

        adj = build_adj_from_tuples(tuples)
        dist = {}
        q = deque()
        for s in starts:
            ss = int(s)
            if ss in dist:
                continue
            dist[ss] = 0
            q.append(ss)

        best = None
        while q:
            cur = q.popleft()
            d = dist[cur]
            if cur in gold:
                best = d
                break
            cand = adj.get(cur, [])
            if neighbor_cap > 0 and len(cand) > neighbor_cap:
                cand = cand[:neighbor_cap]
            for _, nxt in cand:
                nn = int(nxt)
                if nn in dist:
                    continue
                dist[nn] = d + 1
                q.append(nn)

        if best is not None:
            shortest_hops.append(int(best))
            reachable_any += 1
            if best <= steps_cap:
                reachable_at_k += 1
                hop_hist[int(best)] += 1
            else:
                need_more_steps += 1

    def _rate(n: int, d: int) -> float:
        return float(n) / float(max(1, d))

    avg_hops_any = float(sum(shortest_hops)) / float(max(1, len(shortest_hops)))
    out = {
        'total': int(total),
        'usable': int(usable),
        'no_tuple': int(no_tuple),
        'no_start': int(no_start),
        'no_gold': int(no_gold),
        'answer_in_subgraph': int(answer_in_subgraph),
        'answer_in_subgraph_rate_total': _rate(answer_in_subgraph, total),
        'answer_in_subgraph_rate_usable': _rate(answer_in_subgraph, usable),
        'reachable_at_k': int(reachable_at_k),
        'reachable_at_k_rate_total': _rate(reachable_at_k, total),
        'reachable_at_k_rate_usable': _rate(reachable_at_k, usable),
        'reachable_any': int(reachable_any),
        'reachable_any_rate_total': _rate(reachable_any, total),
        'reachable_any_rate_usable': _rate(reachable_any, usable),
        'need_more_steps': int(need_more_steps),
        'need_more_steps_rate_usable': _rate(need_more_steps, usable),
        'avg_shortest_hops_any': float(avg_hops_any),
        'hop_hist_at_k': {str(i): int(hop_hist[i]) for i in range(len(hop_hist))},
        'max_steps': int(steps_cap),
        'start_topk': int(topk),
        'max_neighbors': int(neighbor_cap),
    }
    return out


@torch.no_grad()
def evaluate_relation_beam(
    model,
    carry_init_fn,
    eval_json: str,
    kb2idx: Dict[str, int],
    rel2idx: Dict[str, int],
    device,
    tokenizer_name: str,
    q_npy: str,
    rel_npy: str,
    max_steps: int,
    max_neighbors: int,
    prune_keep: int,
    start_topk: int,
    beam: int,
    max_q_len: int,
    eval_limit: int,
    debug_n: int,
    no_cycle: bool = False,
    eval_pred_topk: int = 1,
    eval_use_halt: bool = False,
    eval_min_hops_before_stop: int = 1,
    entity_labels: Optional[List[str]] = None,
    relation_labels: Optional[List[str]] = None,
):
    model.eval()
    tok = load_tokenizer(tokenizer_name)
    rel_mem = np.load(rel_npy, mmap_mode='r')
    q_mem = np.load(q_npy, mmap_mode='r') if q_npy and os.path.exists(q_npy) else None

    data = list(iter_json_records(eval_json))
    total = len(data) if eval_limit < 0 else min(len(data), eval_limit)
    hit1 = []
    f1s = []
    skip = 0
    debugged = 0
    pred_topk = max(1, int(eval_pred_topk))
    min_hops_before_stop = max(0, int(eval_min_hops_before_stop))

    entity_vocab_size = None
    try:
        inner = model.module.inner if hasattr(model, "module") else model.inner
        emb = getattr(inner, "puzzle_emb_data", None)
        if emb is not None and hasattr(emb, "shape") and len(emb.shape) >= 1:
            entity_vocab_size = int(emb.shape[0])
    except Exception:
        entity_vocab_size = None

    def _safe_entity_id(x: int) -> int:
        i = int(x)
        if i < 0:
            return 0
        if entity_vocab_size is None:
            return i
        return i if i < entity_vocab_size else 0

    def _top_pred_entities(sorted_beams, k: int) -> List[int]:
        out = []
        seen = set()
        for b in sorted_beams:
            n = int(b['nodes'][-1])
            if n in seen:
                continue
            seen.add(n)
            out.append(n)
            if len(out) >= k:
                break
        return out

    for ex_idx in tqdm(range(total), desc='Eval'):
        ex = data[ex_idx]
        tuples, starts_raw, gold = _parse_eval_example(ex, kb2idx, rel2idx)
        adj = build_adj_from_tuples(tuples)
        starts = list(dict.fromkeys(starts_raw))[:max(1, start_topk)]
        if not starts or not tuples:
            skip += 1
            continue

        if q_mem is None:
            q_emb = np.zeros((rel_mem.shape[1],), dtype=np.float32)
        else:
            qi = ex_idx if ex_idx < q_mem.shape[0] else 0
            q_emb = np.asarray(q_mem[qi], dtype=np.float32)

        q_toks = tok(ex.get('question', ''), return_tensors='pt', truncation=True, max_length=max_q_len)
        q_toks = {k: v.to(device) for k, v in q_toks.items()}

        beams = []
        for s in starts:
            init_carry = carry_init_fn({'input_ids': q_toks['input_ids']}, device)
            beams.append({'score': 0.0, 'nodes': [int(s)], 'rels': [], 'carry': init_carry, 'stopped': False})

        beams = beams[:max(1, beam)]

        for _ in range(max_steps):
            new_beams = []
            for b in beams:
                if bool(b.get('stopped', False)):
                    new_beams.append(b)
                    continue
                cur = int(b['nodes'][-1])
                cand = adj.get(cur, [])
                if not cand:
                    nb = dict(b)
                    nb['stopped'] = True
                    new_beams.append(nb)
                    continue
                if max_neighbors > 0 and len(cand) > max_neighbors:
                    cand = cand[:max_neighbors]
                if no_cycle:
                    cand = [(r, nxt) for r, nxt in cand if int(nxt) not in b['nodes']]
                    if not cand:
                        nb = dict(b)
                        nb['stopped'] = True
                        new_beams.append(nb)
                        continue

                rel_cands = []
                seen = set()
                for r, _ in cand:
                    rr = int(r)
                    if rr not in seen:
                        seen.add(rr)
                        rel_cands.append(rr)
                if not rel_cands:
                    nb = dict(b)
                    nb['stopped'] = True
                    new_beams.append(nb)
                    continue
                if prune_keep > 0 and len(rel_cands) > prune_keep:
                    rel_cands = prune_rel_mix_cpu(q_emb, rel_cands, rel_mem, prune_keep, 0)

                cur_pid = _safe_entity_id(cur)
                p_ids = torch.full((1, len(rel_cands)), cur_pid, dtype=torch.long, device=device)
                r_ids = torch.tensor([rel_cands], dtype=torch.long, device=device)
                c_mask = torch.ones_like(r_ids, dtype=torch.bool)
                inp = {
                    'input_ids': q_toks['input_ids'],
                    'attention_mask': q_toks['attention_mask'],
                    'puzzle_identifiers': p_ids,
                    'relation_identifiers': r_ids,
                    'candidate_mask': c_mask,
                }

                next_carry, out = model(b['carry'], inp)
                logp = torch.log_softmax(out['scores'][0], dim=0)
                log_continue = 0.0
                if bool(eval_use_halt) and len(b['rels']) >= min_hops_before_stop:
                    halt_pair = torch.stack(
                        [out['q_continue_logits'][0], out['q_halt_logits'][0]], dim=0
                    )
                    halt_logp = torch.log_softmax(halt_pair, dim=0)
                    log_continue = float(halt_logp[0].item())
                    log_halt = float(halt_logp[1].item())
                    new_beams.append({
                        'score': float(b['score'] + log_halt),
                        'nodes': list(b['nodes']),
                        'rels': list(b['rels']),
                        'carry': _clone_carry(next_carry),
                        'stopped': True,
                    })
                topk = min(max(1, beam), len(rel_cands))
                topv, topi = torch.topk(logp, k=topk)
                for v, i in zip(topv.tolist(), topi.tolist()):
                    r_sel = int(rel_cands[int(i)])
                    next_nodes = []
                    seen_nodes = set()
                    for r, nxt in cand:
                        if int(r) == r_sel and int(nxt) not in seen_nodes:
                            seen_nodes.add(int(nxt))
                            next_nodes.append(int(nxt))
                    if not next_nodes:
                        if no_cycle:
                            continue
                        new_beams.append({
                            'score': float(b['score'] + log_continue + v),
                            'nodes': list(b['nodes']),
                            'rels': list(b['rels']) + [r_sel],
                            'carry': _clone_carry(next_carry),
                            'stopped': False,
                        })
                    else:
                        for nxt in next_nodes:
                            new_beams.append({
                                'score': float(b['score'] + log_continue + v),
                                'nodes': list(b['nodes']) + [int(nxt)],
                                'rels': list(b['rels']) + [r_sel],
                                'carry': _clone_carry(next_carry),
                                'stopped': False,
                            })

            new_beams.sort(key=lambda x: x['score'], reverse=True)
            beams = new_beams[:max(1, beam)]
            if not beams:
                break
            if all(bool(x.get('stopped', False)) for x in beams):
                break

        if not beams:
            skip += 1
            continue

        best = beams[0]
        pred = int(best['nodes'][-1])
        pred_entities = _top_pred_entities(beams, pred_topk)
        pred_set = set(pred_entities)
        if gold:
            h = 1.0 if pred in gold else 0.0
            hit1.append(h)
            inter = len(pred_set & gold)
            if inter == 0:
                f1s.append(0.0)
            else:
                precision = inter / max(1, len(pred_set))
                recall = inter / max(1, len(gold))
                f1s.append((2.0 * precision * recall) / max(1e-12, precision + recall))

        if debugged < max(0, debug_n):
            debugged += 1
            rel_path = ' -> '.join(str(x) for x in best['rels'])
            node_path = ' -> '.join(str(x) for x in best['nodes'])
            print(f"[EvalQ] {ex.get('question', '')}")
            print(f"  relation_path: {rel_path}")
            print(f"  node_path: {node_path}")
            if relation_labels:
                rel_text_path = ' -> '.join(_idx_to_label(relation_labels, x) for x in best['rels'])
                print(f"  relation_text_path: {rel_text_path}")
            if entity_labels:
                node_text_path = ' -> '.join(_idx_to_label(entity_labels, x) for x in best['nodes'])
                print(f"  node_text_path: {node_text_path}")
            print(f"  pred_entity: {pred} | gold_n={len(gold)}")
            print(f"  pred_top{pred_topk}_entities: {pred_entities}")
            if entity_labels:
                pred_top_txt = ' | '.join(_idx_to_label(entity_labels, x) for x in pred_entities)
                print(f"  pred_top{pred_topk}_text: {pred_top_txt}")
            gold_path = _find_gold_path(adj, starts_raw, gold, max_steps=max_steps)
            if gold_path is None:
                print("  gold_path: <not found within max_steps>")
            else:
                g_rels = ' -> '.join(str(x) for x in gold_path["rels"])
                g_nodes = ' -> '.join(str(x) for x in gold_path["nodes"])
                print(f"  gold_relation_path: {g_rels}")
                print(f"  gold_node_path: {g_nodes}")
                if relation_labels:
                    g_rel_txt = ' -> '.join(_idx_to_label(relation_labels, x) for x in gold_path["rels"])
                    print(f"  gold_relation_text_path: {g_rel_txt}")
                if entity_labels:
                    g_node_txt = ' -> '.join(_idx_to_label(entity_labels, x) for x in gold_path["nodes"])
                    print(f"  gold_node_text_path: {g_node_txt}")

    m_hit = float(np.mean(hit1)) if hit1 else 0.0
    m_f1 = float(np.mean(f1s)) if f1s else 0.0
    return m_hit, m_f1, skip


def train(args):
    is_ddp, rank, local_rank, world_size, device = _setup_ddp()
    is_main = rank == 0
    wb = _setup_wandb(args, is_main)

    kb2idx = load_kb_map(args.entities_txt)
    rel2idx = load_rel_map(args.relations_txt)
    entity_labels = _load_entity_labels(args.entities_txt, getattr(args, 'entity_names_json', ''))
    relation_labels = _load_relation_labels(args.relations_txt)
    tok = load_tokenizer(args.trm_tokenizer)
    ent_mem = np.load(args.entity_emb_npy, mmap_mode='r')
    rel_mem = np.load(args.relation_emb_npy, mmap_mode='r')

    cfg = {
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'vocab_size': tok.vocab_size,
        'hidden_size': args.hidden_size,
        'num_heads': args.num_heads,
        'expansion': args.expansion,
        'H_cycles': args.H_cycles,
        'L_cycles': args.L_cycles,
        'L_layers': args.L_layers,
        'H_layers': 0,
        'puzzle_emb_ndim': int(ent_mem.shape[1]),
        'relation_emb_ndim': int(rel_mem.shape[1]),
        'puzzle_emb_len': args.puzzle_emb_len,
        'pos_encodings': args.pos_encodings,
        'forward_dtype': args.forward_dtype,
        'halt_max_steps': args.halt_max_steps,
        'halt_exploration_prob': args.halt_exploration_prob,
        'no_ACT_continue': True,
        'mlp_t': False,
        'num_puzzle_identifiers': len(kb2idx) + 1000,
        'num_relation_identifiers': len(rel2idx) + 1000,
    }

    model, carry_cls = build_model(args.model_impl, args.trm_root, cfg)
    model.inner.puzzle_emb_data = ent_mem
    model.inner.relation_emb_data = rel_mem

    if args.ckpt and os.path.exists(args.ckpt):
        sd = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(sd, strict=False)

    model.to(device)
    if is_ddp:
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=bool(getattr(args, 'ddp_find_unused', False)),
        )

    def carry_init_fn(batch, dev):
        B = int(batch['input_ids'].shape[0])
        inner = model.module.inner if hasattr(model, 'module') else model.inner
        return carry_cls(inner.empty_carry(B), torch.zeros(B, device=dev), torch.ones(B, dtype=torch.bool, device=dev), {})

    if is_main and getattr(args, 'dev_json', '') and bool(getattr(args, 'oracle_diag_enabled', True)):
        diag_max_steps = int(getattr(args, 'eval_max_steps', args.max_steps))
        diag_max_neighbors = int(getattr(args, 'eval_max_neighbors', args.max_neighbors))
        diag_start_topk = int(getattr(args, 'eval_start_topk', getattr(args, 'start_topk', 5)))
        diag_limit = int(getattr(args, 'oracle_diag_limit', -1))
        diag = diagnose_oracle_reachability(
            eval_json=args.dev_json,
            kb2idx=kb2idx,
            rel2idx=rel2idx,
            max_steps=diag_max_steps,
            max_neighbors=diag_max_neighbors,
            start_topk=diag_start_topk,
            eval_limit=diag_limit,
        )
        print(
            "[OracleDiag:Dev] "
            f"usable={diag['usable']}/{diag['total']} "
            f"answer_in_subgraph={diag['answer_in_subgraph_rate_usable']:.4f} "
            f"reachable@{diag_max_steps}={diag['reachable_at_k_rate_usable']:.4f} "
            f"reachable_any={diag['reachable_any_rate_usable']:.4f} "
            f"need_more_steps={diag['need_more_steps_rate_usable']:.4f} "
            f"avg_shortest_hops_any={diag['avg_shortest_hops_any']:.2f}"
        )
        if wb is not None:
            wb.log(
                {
                    "oracle/dev_usable": int(diag['usable']),
                    "oracle/dev_total": int(diag['total']),
                    "oracle/dev_answer_in_subgraph_rate_usable": float(diag['answer_in_subgraph_rate_usable']),
                    "oracle/dev_reachable_at_k_rate_usable": float(diag['reachable_at_k_rate_usable']),
                    "oracle/dev_reachable_any_rate_usable": float(diag['reachable_any_rate_usable']),
                    "oracle/dev_need_more_steps_rate_usable": float(diag['need_more_steps_rate_usable']),
                    "oracle/dev_avg_shortest_hops_any": float(diag['avg_shortest_hops_any']),
                },
                step=0,
            )
        fail_thr = float(getattr(args, 'oracle_diag_fail_threshold', -1.0))
        if fail_thr >= 0.0 and diag['reachable_at_k_rate_usable'] < fail_thr:
            raise RuntimeError(
                f"[OracleDiag:Dev] reachable@{diag_max_steps}={diag['reachable_at_k_rate_usable']:.4f} < threshold={fail_thr:.4f}. "
                "Increase max_steps / neighbors or rebuild preprocessing before training."
            )

    if bool(getattr(args, 'oracle_diag_only', False)):
        if is_ddp:
            dist.barrier()
            dist.destroy_process_group()
        if wb is not None:
            wb.finish()
        return

    ds = PathDataset(args.train_json, args.max_steps)
    endpoint_aux_weight = float(getattr(args, 'endpoint_aux_weight', 0.0))
    collate = make_collate(
        args.trm_tokenizer,
        args.relation_emb_npy,
        args.query_emb_train_npy,
        args.max_neighbors,
        args.prune_keep,
        args.prune_rand,
        args.max_q_len,
        args.max_steps,
        endpoint_aux=endpoint_aux_weight > 0.0,
        entity_vocab_size=int(ent_mem.shape[0]),
    )
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=args.num_workers, drop_last=True, collate_fn=collate, pin_memory=torch.cuda.is_available(), persistent_workers=(args.num_workers > 0))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    bce_endpoint = nn.BCEWithLogitsLoss(reduction='none')
    bce_halt = nn.BCEWithLogitsLoss(reduction='none')
    halt_aux_weight = float(getattr(args, 'halt_aux_weight', 0.0))

    for ep in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(ep)
        model.train()
        pbar = tqdm(loader, disable=not is_main, desc=f'Ep {ep}')
        tot_loss = 0.0
        tot_correct = 0
        tot_count = 0
        steps = 0
        for batch in pbar:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attn = batch['attention_mask'].to(device, non_blocking=True)
            carry = carry_init_fn({'input_ids': input_ids}, device)
            opt.zero_grad(set_to_none=True)
            bl = torch.zeros((), device=device)
            T = len(batch['seq_batches'])
            for t, step in enumerate(batch['seq_batches']):
                p_ids = step['puzzle_identifiers'].to(device)
                r_ids = step['relation_identifiers'].to(device)
                c_mask = step['candidate_mask'].to(device)
                endpoint_t = step['endpoint_targets'].to(device)
                halt_t = step['halt_targets'].to(device)
                halt_m = step['halt_mask'].to(device)
                labels = step['labels'].to(device)
                v_mask = step['valid_mask'].to(device)
                carry, out = model(carry, {'input_ids': input_ids, 'attention_mask': attn, 'puzzle_identifiers': p_ids, 'relation_identifiers': r_ids, 'candidate_mask': c_mask})
                logits = out['scores'].masked_fill(~c_mask, -1e4)
                lv = ce(logits, labels).masked_fill(~v_mask, 0.0)
                valid = v_mask & (labels >= 0)
                if valid.any():
                    pred = torch.argmax(logits, dim=1)
                    tot_correct += int((pred[valid] == labels[valid]).sum().item())
                    tot_count += int(valid.sum().item())
                sc = v_mask.sum().clamp(min=1)
                step_loss = lv.sum() / sc
                if endpoint_aux_weight > 0.0:
                    ev = bce_endpoint(logits, endpoint_t).masked_fill(~c_mask, 0.0)
                    ev = ev.sum(dim=1) / c_mask.sum(dim=1).clamp(min=1).float()
                    ev = ev.masked_fill(~v_mask, 0.0)
                    step_loss = step_loss + endpoint_aux_weight * (ev.sum() / sc)
                if halt_aux_weight > 0.0:
                    halt_logits = out['q_halt_logits'].to(torch.float32)
                    halt_vec = bce_halt(halt_logits, halt_t).masked_fill(~halt_m, 0.0)
                    halt_l = halt_vec.sum() / halt_m.sum().clamp(min=1).float()
                    step_loss = step_loss + halt_aux_weight * halt_l
                bl += ((t + 1) / T) * step_loss
            if not torch.isfinite(bl):
                if is_ddp:
                    raise RuntimeError('non-finite loss in DDP')
                continue
            bl.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += bl.item()
            steps += 1
            if is_main:
                acc = 100.0 * (tot_correct / max(1, tot_count))
                pbar.set_postfix_str(
                    f'[step {steps}] loss={bl.item():.4f} avg={tot_loss/max(1,steps):.4f} '
                    f'acc={acc:.2f}% grad={float(grad_norm):.2e}'
                )
                if wb is not None:
                    wb.log(
                        {
                            "train/step_loss": float(bl.item()),
                            "train/step_avg_loss": float(tot_loss / max(1, steps)),
                            "train/step_acc": float(acc),
                            "train/grad_norm": float(grad_norm),
                            "train/epoch": int(ep),
                            "train/step": int(steps),
                        },
                        step=(ep - 1) * max(1, len(loader)) + steps,
                    )

        if is_ddp:
            dist.barrier()
        if is_main:
            save_obj = model.module if hasattr(model, 'module') else model
            os.makedirs(args.out_dir, exist_ok=True)
            ckpt = os.path.join(args.out_dir, f'model_ep{ep}.pt')
            torch.save(save_obj.state_dict(), ckpt)
            print(f'Saved {ckpt}')
            eval_every = max(1, int(getattr(args, 'eval_every_epochs', 1)))
            eval_start = max(1, int(getattr(args, 'eval_start_epoch', 1)))
            should_eval = ep >= eval_start and ((ep - eval_start) % eval_every == 0)
            if getattr(args, 'dev_json', '') and should_eval:
                # Dev evaluation uses the same endpoint traversal metric as test:
                # start-entity -> predicted end-entity, measured by Hit@1/F1.
                mh, mf, sk = evaluate_relation_beam(
                    model=save_obj,
                    carry_init_fn=carry_init_fn,
                    eval_json=args.dev_json,
                    kb2idx=kb2idx,
                    rel2idx=rel2idx,
                    device=device,
                    tokenizer_name=args.trm_tokenizer,
                    q_npy=getattr(args, 'query_emb_dev_npy', ''),
                    rel_npy=args.relation_emb_npy,
                    max_steps=int(getattr(args, 'eval_max_steps', args.max_steps)),
                    max_neighbors=int(getattr(args, 'eval_max_neighbors', args.max_neighbors)),
                    prune_keep=int(getattr(args, 'eval_prune_keep', args.prune_keep)),
                    start_topk=int(getattr(args, 'eval_start_topk', getattr(args, 'start_topk', 5))),
                    beam=int(getattr(args, 'eval_beam', getattr(args, 'beam', 5))),
                    max_q_len=args.max_q_len,
                    eval_limit=getattr(args, 'eval_limit', -1),
                    debug_n=getattr(args, 'debug_eval_n', 0),
                    no_cycle=bool(getattr(args, 'eval_no_cycle', False)),
                    eval_pred_topk=int(getattr(args, 'eval_pred_topk', 1)),
                    eval_use_halt=bool(getattr(args, 'eval_use_halt', False)),
                    eval_min_hops_before_stop=int(getattr(args, 'eval_min_hops_before_stop', 1)),
                    entity_labels=entity_labels,
                    relation_labels=relation_labels,
                )
                print(f'[Dev] Hit@1={mh:.4f} F1={mf:.4f} Skip={sk}')
                if wb is not None:
                    wb.log(
                        {
                            "dev/hit1": float(mh),
                            "dev/f1": float(mf),
                            "dev/skip": int(sk),
                            "train/epoch": int(ep),
                        },
                        step=ep * max(1, len(loader)),
                    )
            elif getattr(args, 'dev_json', ''):
                print(f'[Dev] skip eval at ep{ep} (start={eval_start}, every={eval_every})')
            if wb is not None:
                wb.log(
                    {
                        "train/epoch_avg_loss": float(tot_loss / max(1, steps)),
                        "train/epoch_acc": float(100.0 * (tot_correct / max(1, tot_count))),
                        "train/epoch": int(ep),
                    },
                    step=ep * max(1, len(loader)),
                )
        # Keep all ranks in sync while rank0 runs (potentially long) dev eval.
        if is_ddp:
            dist.barrier()

    if is_ddp:
        dist.destroy_process_group()
    if wb is not None:
        wb.finish()


def test(args):
    # Full relation-path beam traversal evaluation:
    # start-entity -> predicted end-entity, measured by Hit@1/F1.
    tok = load_tokenizer(args.trm_tokenizer)
    ent_mem = np.load(args.entity_emb_npy, mmap_mode='r')
    rel_mem = np.load(args.relation_emb_npy, mmap_mode='r')
    kb2idx = load_kb_map(args.entities_txt)
    rel2idx = load_rel_map(args.relations_txt)
    entity_labels = _load_entity_labels(args.entities_txt, getattr(args, 'entity_names_json', ''))
    relation_labels = _load_relation_labels(args.relations_txt)
    cfg = {
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'vocab_size': tok.vocab_size,
        'hidden_size': args.hidden_size,
        'num_heads': args.num_heads,
        'expansion': args.expansion,
        'H_cycles': args.H_cycles,
        'L_cycles': args.L_cycles,
        'L_layers': args.L_layers,
        'H_layers': 0,
        'puzzle_emb_ndim': int(ent_mem.shape[1]),
        'relation_emb_ndim': int(rel_mem.shape[1]),
        'puzzle_emb_len': args.puzzle_emb_len,
        'pos_encodings': args.pos_encodings,
        'forward_dtype': args.forward_dtype,
        'halt_max_steps': args.halt_max_steps,
        'halt_exploration_prob': args.halt_exploration_prob,
        'no_ACT_continue': True,
        'mlp_t': False,
        'num_puzzle_identifiers': len(kb2idx) + 1000,
        'num_relation_identifiers': len(rel2idx) + 1000,
    }
    model, _ = build_model(args.model_impl, args.trm_root, cfg)
    model.inner.puzzle_emb_data = ent_mem
    model.inner.relation_emb_data = rel_mem
    sd = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(sd, strict=False)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    dev = next(model.parameters()).device

    carry_cls = type(model.initial_carry({'input_ids': torch.ones((1, 1), dtype=torch.long, device=dev)}))

    def carry_init_fn(batch, device):
        B = int(batch['input_ids'].shape[0])
        inner = model.inner
        return carry_cls(inner.empty_carry(B), torch.zeros(B, device=device), torch.ones(B, dtype=torch.bool, device=device), {})

    print('✅ checkpoint loaded:', args.ckpt)
    if getattr(args, 'eval_json', ''):
        mh, mf, sk = evaluate_relation_beam(
            model=model,
            carry_init_fn=carry_init_fn,
            eval_json=args.eval_json,
            kb2idx=kb2idx,
            rel2idx=rel2idx,
            device=dev,
            tokenizer_name=args.trm_tokenizer,
            q_npy=getattr(args, 'query_emb_eval_npy', ''),
            rel_npy=args.relation_emb_npy,
            max_steps=int(getattr(args, 'eval_max_steps', getattr(args, 'max_steps', 4))),
            max_neighbors=int(getattr(args, 'eval_max_neighbors', getattr(args, 'max_neighbors', 256))),
            prune_keep=int(getattr(args, 'eval_prune_keep', getattr(args, 'prune_keep', 64))),
            start_topk=int(getattr(args, 'eval_start_topk', getattr(args, 'start_topk', 5))),
            beam=int(getattr(args, 'eval_beam', getattr(args, 'beam', 5))),
            max_q_len=getattr(args, 'max_q_len', 512),
            eval_limit=getattr(args, 'eval_limit', -1),
            debug_n=getattr(args, 'debug_eval_n', 5),
            no_cycle=bool(getattr(args, 'eval_no_cycle', False)),
            eval_pred_topk=int(getattr(args, 'eval_pred_topk', 1)),
            eval_use_halt=bool(getattr(args, 'eval_use_halt', False)),
            eval_min_hops_before_stop=int(getattr(args, 'eval_min_hops_before_stop', 1)),
            entity_labels=entity_labels,
            relation_labels=relation_labels,
        )
        print(f'[Test] Hit@1={mh:.4f} F1={mf:.4f} Skip={sk}')
