import importlib
import math
import os
import re
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

from .data import build_adj_from_tuples, iter_json_records, load_kb_map, load_rel_map, read_jsonl_by_offset, build_line_offsets


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
        return {
            'q_text': ex.get('question', ''),
            'tuples': ex.get('subgraph', {}).get('tuples', []),
            'path_segments': segs,
            'ex_line': li,
        }


def make_collate(tokenizer_name, rel_npy, q_npy, max_neighbors, prune_keep, prune_rand, max_q_len, max_steps):
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    rel_mem = np.load(rel_npy, mmap_mode='r')
    q_mem = np.load(q_npy, mmap_mode='r') if q_npy and os.path.exists(q_npy) else None

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
            sample_ctx.append((adj, q_emb))

        seq_batches = []
        for t in range(max_steps):
            puzs, rels, labels, cmask, vmask = [], [], [], [], []
            for i, b in enumerate(batch):
                segs = b['path_segments']
                idx = t * 2
                if idx + 2 >= len(segs):
                    puzs.append([0]); rels.append([0]); labels.append(-100); cmask.append([False]); vmask.append(False)
                    continue
                cur = int(segs[idx])
                gold_r = int(segs[idx + 1])
                adj, q_emb = sample_ctx[i]
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
                puzs.append([0] * len(rel_cands))
                rels.append(rel_cands)
                labels.append(rel_cands.index(gold_r))
                cmask.append([True] * len(rel_cands))
                vmask.append(True)

            cmax = max(len(x) for x in puzs)
            B = len(batch)
            pt = torch.zeros((B, cmax), dtype=torch.long)
            rt = torch.zeros((B, cmax), dtype=torch.long)
            mt = torch.zeros((B, cmax), dtype=torch.bool)
            for k in range(B):
                L = len(puzs[k])
                pt[k, :L] = torch.tensor(puzs[k], dtype=torch.long)
                rt[k, :L] = torch.tensor(rels[k], dtype=torch.long)
                mt[k, :L] = torch.tensor(cmask[k], dtype=torch.bool)
            seq_batches.append({
                'puzzle_identifiers': pt,
                'relation_identifiers': rt,
                'candidate_mask': mt,
                'labels': torch.tensor(labels, dtype=torch.long),
                'valid_mask': torch.tensor(vmask, dtype=torch.bool),
            })

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
):
    model.eval()
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    rel_mem = np.load(rel_npy, mmap_mode='r')
    q_mem = np.load(q_npy, mmap_mode='r') if q_npy and os.path.exists(q_npy) else None

    data = list(iter_json_records(eval_json))
    total = len(data) if eval_limit < 0 else min(len(data), eval_limit)
    hit1 = []
    f1s = []
    skip = 0
    debugged = 0

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

        q_toks = tok(ex.get('question', ''), return_tensors='pt', truncation=True, max_length=max_q_len).to(device)

        beams = []
        for s in starts:
            init_carry = carry_init_fn({'input_ids': q_toks['input_ids']}, device)
            beams.append({'score': 0.0, 'nodes': [int(s)], 'rels': [], 'carry': init_carry})

        beams = beams[:max(1, beam)]

        for _ in range(max_steps):
            new_beams = []
            for b in beams:
                cur = int(b['nodes'][-1])
                cand = adj.get(cur, [])
                if not cand:
                    new_beams.append(b)
                    continue
                if max_neighbors > 0 and len(cand) > max_neighbors:
                    cand = cand[:max_neighbors]

                rel_cands = []
                seen = set()
                for r, _ in cand:
                    rr = int(r)
                    if rr not in seen:
                        seen.add(rr)
                        rel_cands.append(rr)
                if not rel_cands:
                    new_beams.append(b)
                    continue
                if prune_keep > 0 and len(rel_cands) > prune_keep:
                    rel_cands = prune_rel_mix_cpu(q_emb, rel_cands, rel_mem, prune_keep, 0)

                p_ids = torch.zeros((1, len(rel_cands)), dtype=torch.long, device=device)
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
                        new_beams.append({
                            'score': float(b['score'] + v),
                            'nodes': list(b['nodes']),
                            'rels': list(b['rels']) + [r_sel],
                            'carry': _clone_carry(next_carry),
                        })
                    else:
                        for nxt in next_nodes:
                            new_beams.append({
                                'score': float(b['score'] + v),
                                'nodes': list(b['nodes']) + [int(nxt)],
                                'rels': list(b['rels']) + [r_sel],
                                'carry': _clone_carry(next_carry),
                            })

            new_beams.sort(key=lambda x: x['score'], reverse=True)
            beams = new_beams[:max(1, beam)]
            if not beams:
                break

        if not beams:
            skip += 1
            continue

        best = beams[0]
        pred = int(best['nodes'][-1])
        if gold:
            h = 1.0 if pred in gold else 0.0
            hit1.append(h)
            if h > 0:
                p = 1.0
                r = 1.0 / max(1, len(gold))
                f1s.append((2 * p * r) / (p + r))
            else:
                f1s.append(0.0)

        if debugged < max(0, debug_n):
            debugged += 1
            rel_path = ' -> '.join(str(x) for x in best['rels'])
            node_path = ' -> '.join(str(x) for x in best['nodes'])
            print(f"[EvalQ] {ex.get('question', '')}")
            print(f"  relation_path: {rel_path}")
            print(f"  node_path: {node_path}")
            print(f"  pred_entity: {pred} | gold_n={len(gold)}")

    m_hit = float(np.mean(hit1)) if hit1 else 0.0
    m_f1 = float(np.mean(f1s)) if f1s else 0.0
    return m_hit, m_f1, skip


def train(args):
    is_ddp, rank, local_rank, world_size, device = _setup_ddp()
    is_main = rank == 0

    kb2idx = load_kb_map(args.entities_txt)
    rel2idx = load_rel_map(args.relations_txt)
    tok = AutoTokenizer.from_pretrained(args.trm_tokenizer, trust_remote_code=True)
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
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None, output_device=local_rank if torch.cuda.is_available() else None, find_unused_parameters=True)

    def carry_init_fn(batch, dev):
        B = int(batch['input_ids'].shape[0])
        inner = model.module.inner if hasattr(model, 'module') else model.inner
        return carry_cls(inner.empty_carry(B), torch.zeros(B, device=dev), torch.ones(B, dtype=torch.bool, device=dev), {})

    ds = PathDataset(args.train_json, args.max_steps)
    collate = make_collate(args.trm_tokenizer, args.relation_emb_npy, args.query_emb_train_npy, args.max_neighbors, args.prune_keep, args.prune_rand, args.max_q_len, args.max_steps)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=args.num_workers, drop_last=True, collate_fn=collate, pin_memory=torch.cuda.is_available(), persistent_workers=(args.num_workers > 0))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

    for ep in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(ep)
        model.train()
        pbar = tqdm(loader, disable=not is_main, desc=f'Ep {ep}')
        tot_loss = 0.0
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
                labels = step['labels'].to(device)
                v_mask = step['valid_mask'].to(device)
                carry, out = model(carry, {'input_ids': input_ids, 'attention_mask': attn, 'puzzle_identifiers': p_ids, 'relation_identifiers': r_ids, 'candidate_mask': c_mask})
                logits = out['scores'].masked_fill(~c_mask, -1e4)
                lv = ce(logits, labels).masked_fill(~v_mask, 0.0)
                sc = v_mask.sum().clamp(min=1)
                bl += ((t + 1) / T) * (lv.sum() / sc)
            if not torch.isfinite(bl):
                if is_ddp:
                    raise RuntimeError('non-finite loss in DDP')
                continue
            bl.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += bl.item()
            steps += 1
            if is_main:
                pbar.set_postfix_str(f'loss={bl.item():.4f} avg={tot_loss/max(1,steps):.4f}')

        if is_ddp:
            dist.barrier()
        if is_main:
            save_obj = model.module if hasattr(model, 'module') else model
            os.makedirs(args.out_dir, exist_ok=True)
            ckpt = os.path.join(args.out_dir, f'model_ep{ep}.pt')
            torch.save(save_obj.state_dict(), ckpt)
            print(f'Saved {ckpt}')
            if getattr(args, 'dev_json', ''):
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
                    max_steps=args.max_steps,
                    max_neighbors=args.max_neighbors,
                    prune_keep=args.prune_keep,
                    start_topk=getattr(args, 'start_topk', 5),
                    beam=getattr(args, 'beam', 5),
                    max_q_len=args.max_q_len,
                    eval_limit=getattr(args, 'eval_limit', -1),
                    debug_n=getattr(args, 'debug_eval_n', 0),
                )
                print(f'[Dev] Hit@1={mh:.4f} F1={mf:.4f} Skip={sk}')

    if is_ddp:
        dist.destroy_process_group()


def test(args):
    # Full relation-path beam traversal evaluation (same multi-hop behavior as test-time search).
    tok = AutoTokenizer.from_pretrained(args.trm_tokenizer, trust_remote_code=True)
    ent_mem = np.load(args.entity_emb_npy, mmap_mode='r')
    rel_mem = np.load(args.relation_emb_npy, mmap_mode='r')
    kb2idx = load_kb_map(args.entities_txt)
    rel2idx = load_rel_map(args.relations_txt)
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
            max_steps=getattr(args, 'max_steps', 4),
            max_neighbors=getattr(args, 'max_neighbors', 256),
            prune_keep=getattr(args, 'prune_keep', 64),
            start_topk=getattr(args, 'start_topk', 5),
            beam=getattr(args, 'beam', 5),
            max_q_len=getattr(args, 'max_q_len', 512),
            eval_limit=getattr(args, 'eval_limit', -1),
            debug_n=getattr(args, 'debug_eval_n', 5),
        )
        print(f'[Test] Hit@1={mh:.4f} F1={mf:.4f} Skip={sk}')
