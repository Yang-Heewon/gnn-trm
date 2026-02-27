import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .data import build_line_offsets, load_rel_map, read_jsonl_by_offset


def _as_bool(v, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return bool(default)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _coerce_int_list(values) -> List[int]:
    out = []
    for x in values or []:
        try:
            out.append(int(x))
        except Exception:
            continue
    return out


def _parse_tuples(raw_tuples, rel2idx: Optional[Dict[str, int]] = None) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    for tri in raw_tuples or []:
        if not isinstance(tri, (list, tuple)) or len(tri) != 3:
            continue
        s, r, o = tri
        try:
            ss = int(s)
            oo = int(o)
        except Exception:
            continue
        if isinstance(r, int):
            rr = int(r)
        elif isinstance(r, str) and rel2idx is not None and r in rel2idx:
            rr = int(rel2idx[r])
        else:
            continue
        out.append((ss, rr, oo))
    return out


def _extract_seed_entities(ex: dict, tuples: List[Tuple[int, int, int]]) -> List[int]:
    starts = _coerce_int_list(ex.get("entities_cid", []))
    if not starts:
        starts = _coerce_int_list(ex.get("entities", []))
    if not starts and tuples:
        starts = [int(tuples[0][0])]
    return list(dict.fromkeys(starts))


def _extract_gold_answers(ex: dict) -> List[int]:
    return list(dict.fromkeys(_coerce_int_list(ex.get("answers_cid", []))))


def _build_khop_subgraph(
    tuples: List[Tuple[int, int, int]],
    seed_nodes: Sequence[int],
    hops: int,
    max_nodes: int,
    max_edges: int,
    add_reverse_edges: bool,
    split_reverse_relations: bool = False,
    relation_offset: int = 0,
) -> Tuple[List[int], List[Tuple[int, int, int]]]:
    hops = max(0, int(hops))
    max_nodes = max(1, int(max_nodes))
    max_edges = max(0, int(max_edges))

    if not tuples:
        if seed_nodes:
            return [int(seed_nodes[0])], []
        return [0], []

    all_edges: List[Tuple[int, int, int]] = []
    adj: Dict[int, List[Tuple[int, int]]] = {}
    for s, r, o in tuples:
        ss = int(s)
        rr = int(r)
        oo = int(o)
        all_edges.append((ss, rr, oo))
        adj.setdefault(ss, []).append((rr, oo))
        if add_reverse_edges:
            rev_rr = int(rr + relation_offset) if split_reverse_relations else rr
            all_edges.append((oo, rev_rr, ss))
            adj.setdefault(oo, []).append((rev_rr, ss))

    seeds = [int(x) for x in seed_nodes if isinstance(x, (int, np.integer))]
    if not seeds:
        seeds = [int(tuples[0][0])]

    visited = set()
    node_order: List[int] = []
    frontier: List[int] = []
    for s in seeds:
        if s in visited:
            continue
        visited.add(s)
        node_order.append(s)
        frontier.append(s)
        if len(node_order) >= max_nodes:
            break

    traversed: List[Tuple[int, int, int]] = []
    traversed_set = set()
    for _ in range(hops):
        if not frontier:
            break
        new_frontier: List[int] = []
        for u in frontier:
            for rr, vv in adj.get(int(u), []):
                tri = (int(u), int(rr), int(vv))
                if tri not in traversed_set and (max_edges <= 0 or len(traversed) < max_edges):
                    traversed.append(tri)
                    traversed_set.add(tri)
                if vv in visited:
                    continue
                if len(node_order) >= max_nodes:
                    continue
                visited.add(int(vv))
                node_order.append(int(vv))
                new_frontier.append(int(vv))
        frontier = new_frontier

    if max_edges > 0 and len(traversed) < max_edges:
        for s, r, o in all_edges:
            if s not in visited or o not in visited:
                continue
            tri = (int(s), int(r), int(o))
            if tri in traversed_set:
                continue
            traversed.append(tri)
            traversed_set.add(tri)
            if len(traversed) >= max_edges:
                break

    if not node_order:
        node_order = [int(seeds[0])] if seeds else [0]

    node2idx = {n: i for i, n in enumerate(node_order)}
    edges_idx: List[Tuple[int, int, int]] = []
    for s, r, o in traversed:
        if s not in node2idx or o not in node2idx:
            continue
        edges_idx.append((node2idx[s], node2idx[o], int(r)))

    return node_order, edges_idx


class SubgraphExampleDataset(Dataset):
    def __init__(self, jsonl_path: str):
        if not jsonl_path:
            raise RuntimeError("subgraph reader requires a jsonl path.")
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"jsonl not found: {jsonl_path}")
        self.jsonl_path = jsonl_path
        self.offsets = build_line_offsets(jsonl_path, is_main=True)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> dict:
        ex = read_jsonl_by_offset(self.jsonl_path, self.offsets, idx)
        return {
            "orig_id": ex.get("orig_id", ex.get("id", str(idx))),
            "question": ex.get("question", ""),
            "tuples": ex.get("subgraph", {}).get("tuples", []),
            "entities": ex.get("entities", []),
            "entities_cid": ex.get("entities_cid", []),
            "answers_cid": ex.get("answers_cid", []),
            "ex_line": idx,
        }


class SubgraphCollator:
    def __init__(
        self,
        entity_emb_npy: str,
        relation_emb_npy: str,
        query_emb_npy: str,
        rel2idx: Optional[Dict[str, int]],
        hops: int = 3,
        max_nodes: int = 256,
        max_edges: int = 2048,
        add_reverse_edges: bool = True,
        split_reverse_relations: bool = False,
    ):
        if not query_emb_npy:
            raise RuntimeError("query embedding path is empty for subgraph reader.")
        if not os.path.exists(query_emb_npy):
            raise FileNotFoundError(f"query embedding file not found: {query_emb_npy}")
        self.ent_mem = np.load(entity_emb_npy, mmap_mode="r")
        self.rel_mem = np.load(relation_emb_npy, mmap_mode="r")
        self.q_mem = np.load(query_emb_npy, mmap_mode="r")
        self.rel2idx = rel2idx
        self.hops = max(0, int(hops))
        self.max_nodes = max(1, int(max_nodes))
        self.max_edges = max(0, int(max_edges))
        self.add_reverse_edges = bool(add_reverse_edges)
        self.split_reverse_relations = bool(split_reverse_relations)
        self.entity_dim = int(self.ent_mem.shape[1])
        self.relation_dim = int(self.rel_mem.shape[1])
        self.query_dim = int(self.q_mem.shape[1])
        self.num_relations = int(self.rel_mem.shape[0])

    def _gather_entity_emb(self, node_ids: List[int]) -> np.ndarray:
        out = np.zeros((len(node_ids), self.entity_dim), dtype=np.float32)
        for i, nid in enumerate(node_ids):
            if 0 <= int(nid) < int(self.ent_mem.shape[0]):
                out[i] = np.asarray(self.ent_mem[int(nid)], dtype=np.float32)
        return out

    def _gather_relation_emb(self, rel_ids: List[int]) -> np.ndarray:
        out = np.zeros((len(rel_ids), self.relation_dim), dtype=np.float32)
        for i, rid in enumerate(rel_ids):
            if 0 <= int(rid) < int(self.rel_mem.shape[0]):
                out[i] = np.asarray(self.rel_mem[int(rid)], dtype=np.float32)
        return out

    def __call__(self, batch: List[dict]) -> dict:
        node_emb_list = []
        node_mask_list = []
        seed_mask_list = []
        node_label_list = []
        node_cids_list = []
        edge_src_list = []
        edge_dst_list = []
        edge_rel_emb_list = []
        edge_dir_list = []
        edge_mask_list = []
        q_emb_list = []
        gold_list = []
        orig_ids = []

        max_n = 1
        max_e = 0
        for ex in batch:
            tuples = _parse_tuples(ex.get("tuples", []), self.rel2idx)
            seeds = _extract_seed_entities(ex, tuples)
            gold = _extract_gold_answers(ex)
            node_cids, edges_idx = _build_khop_subgraph(
                tuples=tuples,
                seed_nodes=seeds,
                hops=self.hops,
                max_nodes=self.max_nodes,
                max_edges=self.max_edges,
                add_reverse_edges=self.add_reverse_edges,
                split_reverse_relations=self.split_reverse_relations,
                relation_offset=self.num_relations,
            )
            if not node_cids:
                node_cids = [0]
            n = len(node_cids)
            e = len(edges_idx)
            max_n = max(max_n, n)
            max_e = max(max_e, e)

            node_emb = self._gather_entity_emb(node_cids)
            seed_set = set(int(x) for x in seeds)
            seed_mask = np.asarray([int(nid) in seed_set for nid in node_cids], dtype=np.bool_)
            # Fallback for malformed examples where no seed survives subgraph crop.
            if not bool(seed_mask.any()) and n > 0:
                seed_mask[0] = True
            gold_set = set(int(x) for x in gold)
            node_lbl = np.asarray([1.0 if int(nid) in gold_set else 0.0 for nid in node_cids], dtype=np.float32)

            edge_src = np.zeros((e,), dtype=np.int64)
            edge_dst = np.zeros((e,), dtype=np.int64)
            edge_dir = np.zeros((e,), dtype=np.int64)
            edge_rel_ids = []
            for j, (sidx, didx, rid) in enumerate(edges_idx):
                edge_src[j] = int(sidx)
                edge_dst[j] = int(didx)
                rr = int(rid)
                if self.split_reverse_relations and rr >= self.num_relations:
                    rr = rr - self.num_relations
                    edge_dir[j] = 1
                rr = int(rr % max(1, self.num_relations))
                edge_rel_ids.append(rr)
            edge_rel_emb = self._gather_relation_emb(edge_rel_ids)

            qi = int(ex.get("ex_line", -1))
            if qi < 0 or qi >= int(self.q_mem.shape[0]):
                raise RuntimeError(
                    f"query embedding index out of range for subgraph reader: ex_line={qi}, "
                    f"q_rows={int(self.q_mem.shape[0])}"
                )
            q_emb = np.asarray(self.q_mem[qi], dtype=np.float32).copy()

            node_emb_list.append(node_emb)
            node_mask_list.append(np.ones((n,), dtype=np.bool_))
            seed_mask_list.append(seed_mask)
            node_label_list.append(node_lbl)
            node_cids_list.append(np.asarray(node_cids, dtype=np.int64))
            edge_src_list.append(edge_src)
            edge_dst_list.append(edge_dst)
            edge_rel_emb_list.append(edge_rel_emb)
            edge_dir_list.append(edge_dir)
            edge_mask_list.append(np.ones((e,), dtype=np.bool_))
            q_emb_list.append(q_emb)
            gold_list.append(gold)
            orig_ids.append(str(ex.get("orig_id", "")))

        bsz = len(batch)
        node_emb_t = torch.zeros((bsz, max_n, self.entity_dim), dtype=torch.float32)
        node_mask_t = torch.zeros((bsz, max_n), dtype=torch.bool)
        seed_mask_t = torch.zeros((bsz, max_n), dtype=torch.bool)
        node_label_t = torch.zeros((bsz, max_n), dtype=torch.float32)
        node_cids_t = torch.full((bsz, max_n), -1, dtype=torch.long)

        edge_src_t = torch.zeros((bsz, max_e), dtype=torch.long)
        edge_dst_t = torch.zeros((bsz, max_e), dtype=torch.long)
        edge_rel_emb_t = torch.zeros((bsz, max_e, self.relation_dim), dtype=torch.float32)
        edge_dir_t = torch.zeros((bsz, max_e), dtype=torch.long)
        edge_mask_t = torch.zeros((bsz, max_e), dtype=torch.bool)

        q_emb_t = torch.zeros((bsz, self.query_dim), dtype=torch.float32)

        for i in range(bsz):
            n = node_emb_list[i].shape[0]
            e = edge_rel_emb_list[i].shape[0]
            node_emb_t[i, :n] = torch.from_numpy(node_emb_list[i])
            node_mask_t[i, :n] = torch.from_numpy(node_mask_list[i])
            seed_mask_t[i, :n] = torch.from_numpy(seed_mask_list[i])
            node_label_t[i, :n] = torch.from_numpy(node_label_list[i])
            node_cids_t[i, :n] = torch.from_numpy(node_cids_list[i])
            if e > 0:
                edge_src_t[i, :e] = torch.from_numpy(edge_src_list[i])
                edge_dst_t[i, :e] = torch.from_numpy(edge_dst_list[i])
                edge_rel_emb_t[i, :e] = torch.from_numpy(edge_rel_emb_list[i])
                edge_dir_t[i, :e] = torch.from_numpy(edge_dir_list[i])
                edge_mask_t[i, :e] = torch.from_numpy(edge_mask_list[i])
            q_emb_t[i] = torch.from_numpy(q_emb_list[i])

        return {
            "node_emb": node_emb_t,
            "node_mask": node_mask_t,
            "seed_mask": seed_mask_t,
            "node_labels": node_label_t,
            "node_cids": node_cids_t,
            "edge_src": edge_src_t,
            "edge_dst": edge_dst_t,
            "edge_rel_emb": edge_rel_emb_t,
            "edge_dir": edge_dir_t,
            "edge_mask": edge_mask_t,
            "q_emb": q_emb_t,
            "gold_answers": gold_list,
            "orig_ids": orig_ids,
        }


class RecursiveSubgraphReader(nn.Module):
    def __init__(
        self,
        entity_dim: int,
        relation_dim: int,
        query_dim: int,
        hidden_size: int,
        recursion_steps: int = 8,
        dropout: float = 0.1,
        use_direction_embedding: bool = False,
        outer_reasoning_enabled: bool = False,
        outer_reasoning_steps: int = 3,
        gnn_variant: str = "rearev_bfs",
        rearev_num_instructions: int = 3,
        rearev_adapt_stages: int = 1,
    ):
        super().__init__()
        self.entity_dim = int(entity_dim)
        self.relation_dim = int(relation_dim)
        self.query_dim = int(query_dim)
        self.hidden_size = int(hidden_size)
        self.recursion_steps = max(1, int(recursion_steps))
        self.use_direction_embedding = bool(use_direction_embedding)
        self.outer_reasoning_enabled = bool(outer_reasoning_enabled)
        self.outer_reasoning_steps = max(1, int(outer_reasoning_steps))
        variant = str(gnn_variant or "rearev_bfs").strip().lower()
        if variant != "rearev_bfs":
            raise ValueError(
                f"Only rearev_bfs is supported in this build, got gnn_variant={variant!r}."
            )
        self.gnn_variant = variant
        self.rearev_num_instructions = max(1, int(rearev_num_instructions))
        self.rearev_adapt_stages = max(1, int(rearev_adapt_stages))

        self.node_proj = nn.Linear(self.entity_dim, self.hidden_size)
        self.rel_proj = nn.Linear(self.relation_dim, self.hidden_size)
        self.q_proj = nn.Linear(self.query_dim, self.hidden_size)
        # Baseline branch is intentionally disabled in this build.
        self.msg_mlp = None
        self.cell = None
        self.step_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(float(dropout))
        self.out_head = nn.Linear(self.hidden_size, 1)
        self.rearev_ins_proj = nn.Linear(self.hidden_size, self.hidden_size * self.rearev_num_instructions)
        self.rearev_fuse = nn.Sequential(
            nn.Linear(self.hidden_size * (1 + self.rearev_num_instructions), self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.rearev_cell = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.rearev_node_score = nn.Linear(self.hidden_size, 1)
        self.rearev_ins_update = nn.GRUCell(self.hidden_size, self.hidden_size)
        if self.outer_reasoning_enabled:
            self.outer_state_update = nn.GRUCell(self.hidden_size, self.hidden_size)
            self.outer_state_norm = nn.LayerNorm(self.hidden_size)
        if self.use_direction_embedding:
            self.edge_dir_emb = nn.Embedding(2, self.hidden_size)

    def _inner_recur(
        self,
        h: torch.Tensor,
        h0: torch.Tensor,
        src: Optional[torch.Tensor],
        dst: Optional[torch.Tensor],
        rel_h: Optional[torch.Tensor],
        ones: Optional[torch.Tensor],
        q_inj: torch.Tensor,
        n: int,
        e: int,
    ) -> torch.Tensor:
        raise RuntimeError("baseline recurrence is disabled; use gnn_variant='rearev_bfs'.")

    def _seed_distribution(
        self,
        seed_mask: Optional[torch.Tensor],
        n: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if seed_mask is None:
            p = torch.zeros((n,), dtype=dtype, device=device)
            p[0] = 1.0
            return p
        sm = seed_mask[:n].to(device=device)
        p = sm.to(dtype=dtype)
        denom = p.sum()
        if float(denom.item()) <= 0.0:
            p = torch.zeros((n,), dtype=dtype, device=device)
            p[0] = 1.0
            return p
        return p / denom

    def _inner_recur_rearev(
        self,
        h: torch.Tensor,
        h0: torch.Tensor,
        src: Optional[torch.Tensor],
        dst: Optional[torch.Tensor],
        rel_h: Optional[torch.Tensor],
        q_inj: torch.Tensor,
        seed_mask: Optional[torch.Tensor],
        n: int,
        e: int,
    ) -> torch.Tensor:
        if (
            self.rearev_ins_proj is None
            or self.rearev_fuse is None
            or self.rearev_cell is None
            or self.rearev_node_score is None
            or self.rearev_ins_update is None
        ):
            raise RuntimeError("rearev modules are not initialized for this gnn_variant.")
        q_state = q_inj.squeeze(0)
        ins = self.rearev_ins_proj(q_state).view(self.rearev_num_instructions, self.hidden_size)
        h_stage = h
        p_stage = self._seed_distribution(seed_mask=seed_mask, n=n, dtype=h.dtype, device=h.device)

        for stage_idx in range(self.rearev_adapt_stages):
            for _ in range(self.recursion_steps):
                agg_per_ins: List[torch.Tensor] = []
                if e > 0 and src is not None and dst is not None and rel_h is not None:
                    src_prob = p_stage[src].unsqueeze(-1)
                    for k in range(self.rearev_num_instructions):
                        rel_cond = F.relu(rel_h * ins[k].unsqueeze(0))
                        weighted = src_prob * rel_cond
                        agg_k = torch.zeros_like(h_stage)
                        agg_k.index_add_(0, dst, weighted)
                        agg_per_ins.append(agg_k)
                else:
                    for _ in range(self.rearev_num_instructions):
                        agg_per_ins.append(torch.zeros_like(h_stage))

                fuse_in = torch.cat([h_stage] + agg_per_ins, dim=-1)
                agg = self.rearev_fuse(fuse_in) + 0.1 * h0
                h_stage = self.rearev_cell(agg, h_stage)
                h_stage = self.step_norm(h_stage)
                h_stage = self.dropout(h_stage)
                p_stage = torch.softmax(self.rearev_node_score(h_stage).squeeze(-1), dim=0)

            if stage_idx + 1 >= self.rearev_adapt_stages:
                break

            if seed_mask is not None and bool(seed_mask[:n].any()):
                he = h_stage[seed_mask[:n].to(device=h_stage.device)].mean(dim=0)
            else:
                he = h_stage.mean(dim=0)
            he_rep = he.unsqueeze(0).expand(self.rearev_num_instructions, -1)
            ins = self.rearev_ins_update(he_rep, ins)
            # ReaRev-like stage reset: restart propagation from seed distribution.
            p_stage = self._seed_distribution(seed_mask=seed_mask, n=n, dtype=h.dtype, device=h.device)

        return h_stage

    def _run_inner(
        self,
        h: torch.Tensor,
        h0: torch.Tensor,
        src: Optional[torch.Tensor],
        dst: Optional[torch.Tensor],
        rel_h: Optional[torch.Tensor],
        ones: Optional[torch.Tensor],
        q_inj: torch.Tensor,
        seed_mask: Optional[torch.Tensor],
        n: int,
        e: int,
    ) -> torch.Tensor:
        return self._inner_recur_rearev(
            h=h,
            h0=h0,
            src=src,
            dst=dst,
            rel_h=rel_h,
            q_inj=q_inj,
            seed_mask=seed_mask,
            n=n,
            e=e,
        )

    def forward(
        self,
        node_emb: torch.Tensor,
        node_mask: torch.Tensor,
        seed_mask: Optional[torch.Tensor],
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rel_emb: torch.Tensor,
        edge_dir: Optional[torch.Tensor],
        edge_mask: torch.Tensor,
        q_emb: torch.Tensor,
    ) -> torch.Tensor:
        bsz, max_n, _ = node_emb.shape
        qh = self.q_proj(q_emb)
        h_all = self.node_proj(node_emb) + qh.unsqueeze(1)
        logits = node_emb.new_full((bsz, max_n), -1e4)

        for i in range(bsz):
            n = int(node_mask[i].sum().item())
            if n <= 0:
                continue
            h = h_all[i, :n]
            h0 = h.clone()
            qb = qh[i].unsqueeze(0)
            seed_i = seed_mask[i, :n] if seed_mask is not None else None

            e = int(edge_mask[i].sum().item()) if edge_mask.shape[1] > 0 else 0
            if e > 0:
                src = edge_src[i, :e].long()
                dst = edge_dst[i, :e].long()
                rel_h = self.rel_proj(edge_rel_emb[i, :e])
                if self.use_direction_embedding and edge_dir is not None:
                    dir_ids = edge_dir[i, :e].long().clamp(min=0, max=1)
                    rel_h = rel_h + self.edge_dir_emb(dir_ids)
                ones = torch.ones((e,), dtype=h.dtype, device=h.device)
            else:
                src = None
                dst = None
                rel_h = None
                ones = None

            if not self.outer_reasoning_enabled:
                h = self._run_inner(
                    h=h,
                    h0=h0,
                    src=src,
                    dst=dst,
                    rel_h=rel_h,
                    ones=ones,
                    q_inj=qb,
                    seed_mask=seed_i,
                    n=n,
                    e=e,
                )
                y_logits = self.out_head(h).squeeze(-1)
            else:
                # Global latent recursion over (y_k, z_k):
                # y_k: temporary node-answer distribution (from node logits)
                # z_k: global reasoning state updated from y_k-weighted graph context
                z = qb.squeeze(0)
                y_logits = self.out_head(h).squeeze(-1)
                for _ in range(self.outer_reasoning_steps):
                    y_prob = torch.sigmoid(y_logits)
                    denom = y_prob.sum().clamp(min=1e-6)
                    ctx = (h * y_prob.unsqueeze(-1)).sum(dim=0) / denom
                    z = self.outer_state_update(ctx.unsqueeze(0), z.unsqueeze(0)).squeeze(0)
                    z = self.outer_state_norm(z)
                    q_loop = qb + z.unsqueeze(0)
                    h = self._run_inner(
                        h=h,
                        h0=h0,
                        src=src,
                        dst=dst,
                        rel_h=rel_h,
                        ones=ones,
                        q_inj=q_loop,
                        seed_mask=seed_i,
                        n=n,
                        e=e,
                    )
                    y_logits = self.out_head(h).squeeze(-1)

            logits[i, :n] = y_logits

        return logits


def _masked_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    loss = loss * mask.to(torch.float32)
    denom = mask.to(torch.float32).sum().clamp(min=1.0)
    return loss.sum() / denom


def _masked_bce_hard_negative_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    hard_negative_enabled: bool = False,
    hard_negative_topk: int = 64,
) -> Tuple[torch.Tensor, int]:
    raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    if (not hard_negative_enabled) or int(hard_negative_topk) <= 0:
        weighted = raw * mask.to(torch.float32)
        denom = mask.to(torch.float32).sum().clamp(min=1.0)
        return weighted.sum() / denom, int(mask.to(torch.int64).sum().item())

    bsz = int(logits.shape[0])
    keep_mask = torch.zeros_like(mask, dtype=torch.bool)
    total_keep = 0
    topk = max(1, int(hard_negative_topk))

    with torch.no_grad():
        for i in range(bsz):
            valid = torch.nonzero(mask[i], as_tuple=False).squeeze(-1)
            if valid.numel() <= 0:
                continue
            y = targets[i, valid]
            pos = valid[y > 0.5]
            neg = valid[y <= 0.5]

            row_keep = torch.zeros_like(mask[i], dtype=torch.bool)
            if pos.numel() > 0:
                row_keep[pos] = True
            if neg.numel() > 0:
                k = min(topk, int(neg.numel()))
                hard = torch.topk(logits[i, neg].detach(), k=k, largest=True).indices
                hard_neg = neg[hard]
                row_keep[hard_neg] = True
            if not row_keep.any():
                row_keep[valid] = True
            keep_mask[i] = row_keep
            total_keep += int(row_keep.sum().item())

    weighted = raw * keep_mask.to(torch.float32)
    denom = keep_mask.to(torch.float32).sum().clamp(min=1.0)
    return weighted.sum() / denom, int(total_keep)


def _ranking_hard_negative_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    margin: float = 0.2,
    hard_negative_topk: int = 16,
) -> Tuple[torch.Tensor, int]:
    bsz = int(logits.shape[0])
    losses: List[torch.Tensor] = []
    pair_count = 0
    topk = max(1, int(hard_negative_topk))
    m = float(margin)

    for i in range(bsz):
        valid = torch.nonzero(mask[i], as_tuple=False).squeeze(-1)
        if valid.numel() <= 0:
            continue
        y = targets[i, valid]
        pos = valid[y > 0.5]
        neg = valid[y <= 0.5]
        if pos.numel() <= 0 or neg.numel() <= 0:
            continue

        pos_score = logits[i, pos].min()
        neg_scores = logits[i, neg]
        k = min(topk, int(neg_scores.numel()))
        hard_neg = torch.topk(neg_scores, k=k, largest=True).values
        losses.append(F.relu(m - pos_score + hard_neg).mean())
        pair_count += int(k)

    if not losses:
        return logits.new_tensor(0.0), 0
    return torch.stack(losses).mean(), int(pair_count)


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _sample_metrics_from_logits(
    logits_row: torch.Tensor,
    mask_row: torch.Tensor,
    node_cids_row: torch.Tensor,
    gold_answers: Sequence[int],
    pred_topk: int,
    threshold: float,
) -> Tuple[Optional[float], Optional[float]]:
    gold = set(int(x) for x in gold_answers)
    if not gold:
        return None, None

    valid = torch.nonzero(mask_row, as_tuple=False).squeeze(-1)
    if valid.numel() <= 0:
        return None, None

    vals = torch.sigmoid(logits_row[valid])
    cids = node_cids_row[valid]
    keep = torch.nonzero(cids >= 0, as_tuple=False).squeeze(-1)
    if keep.numel() <= 0:
        return None, None
    vals = vals[keep]
    cids = cids[keep]

    order = torch.argsort(vals, descending=True)
    k = max(1, int(pred_topk))
    top_idx = order[: min(k, int(order.numel()))]
    top_nodes = [int(cids[j].item()) for j in top_idx]
    hit1 = 1.0 if top_nodes and int(top_nodes[0]) in gold else 0.0

    pred_set = set()
    thr = float(threshold)
    if thr > 0.0:
        sel = torch.nonzero(vals >= thr, as_tuple=False).squeeze(-1)
        pred_set = {int(cids[j].item()) for j in sel.tolist()}
    if not pred_set:
        pred_set = set(top_nodes)

    inter = len(pred_set & gold)
    precision = float(inter) / float(max(1, len(pred_set)))
    recall = float(inter) / float(max(1, len(gold)))
    f1 = 0.0
    if (precision + recall) > 0.0:
        f1 = (2.0 * precision * recall) / (precision + recall)
    return float(hit1), float(f1)


@torch.no_grad()
def evaluate_subgraph_reader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pred_topk: int,
    threshold: float,
    is_main: bool,
    desc: str = "Eval-Subgraph",
    return_counts: bool = False,
) -> Tuple[float, float, int]:
    model.eval()
    hit_sum = 0.0
    f1_sum = 0.0
    n_valid = 0
    skip = 0

    pbar = tqdm(loader, disable=not is_main, desc=desc)
    for batch in pbar:
        meta_gold = batch["gold_answers"]
        batch_dev = _move_batch_to_device(batch, device)
        logits = model(
            node_emb=batch_dev["node_emb"],
            node_mask=batch_dev["node_mask"],
            seed_mask=batch_dev.get("seed_mask", None),
            edge_src=batch_dev["edge_src"],
            edge_dst=batch_dev["edge_dst"],
            edge_rel_emb=batch_dev["edge_rel_emb"],
            edge_dir=batch_dev.get("edge_dir", None),
            edge_mask=batch_dev["edge_mask"],
            q_emb=batch_dev["q_emb"],
        )
        bsz = int(logits.shape[0])
        for i in range(bsz):
            hit, f1 = _sample_metrics_from_logits(
                logits_row=logits[i],
                mask_row=batch_dev["node_mask"][i],
                node_cids_row=batch_dev["node_cids"][i],
                gold_answers=meta_gold[i],
                pred_topk=pred_topk,
                threshold=threshold,
            )
            if hit is None or f1 is None:
                skip += 1
                continue
            hit_sum += float(hit)
            f1_sum += float(f1)
            n_valid += 1

    mean_hit = float(hit_sum / max(1, n_valid))
    mean_f1 = float(f1_sum / max(1, n_valid))
    if return_counts:
        return mean_hit, mean_f1, int(skip), int(n_valid)
    return mean_hit, mean_f1, int(skip)


def _safe_load_state_dict(model: nn.Module, sd: dict) -> Tuple[int, int]:
    if not isinstance(sd, dict):
        return 0, 0
    cur = model.state_dict()
    keep = {}
    skipped = 0
    for k, v in sd.items():
        if k not in cur:
            continue
        tgt = cur[k]
        try:
            same_shape = tuple(v.shape) == tuple(tgt.shape)
        except Exception:
            same_shape = False
        if same_shape:
            keep[k] = v
        else:
            skipped += 1
    missing, _ = model.load_state_dict(keep, strict=False)
    return int(skipped), int(len(missing))


def _infer_resume_epoch_from_ckpt(ckpt_path: str) -> int:
    if not ckpt_path:
        return 0
    base = os.path.basename(str(ckpt_path))
    m = re.search(r"model_ep(\d+)\.pt$", base)
    if not m:
        return 0
    try:
        return max(0, int(m.group(1)))
    except Exception:
        return 0


def _load_model_from_ckpt_or_init(
    ckpt_path: str,
    entity_dim: int,
    relation_dim: int,
    query_dim: int,
    hidden_size: int,
    recursion_steps: int,
    dropout: float,
    use_direction_embedding: bool = False,
    outer_reasoning_enabled: bool = False,
    outer_reasoning_steps: int = 3,
    gnn_variant: str = "rearev_bfs",
    rearev_num_instructions: int = 3,
    rearev_adapt_stages: int = 1,
) -> Tuple[RecursiveSubgraphReader, Optional[dict]]:
    meta = None
    if ckpt_path and os.path.exists(ckpt_path):
        obj = torch.load(ckpt_path, map_location="cpu")
        if isinstance(obj, dict):
            meta = obj
            model_cfg = obj.get("model_cfg", {})
            if isinstance(model_cfg, dict) and "use_direction_embedding" in model_cfg:
                use_direction_embedding = _as_bool(model_cfg.get("use_direction_embedding", False))
            if isinstance(model_cfg, dict) and "outer_reasoning_enabled" in model_cfg:
                outer_reasoning_enabled = _as_bool(model_cfg.get("outer_reasoning_enabled", False))
            if isinstance(model_cfg, dict) and "outer_reasoning_steps" in model_cfg:
                outer_reasoning_steps = int(model_cfg.get("outer_reasoning_steps", 3))
            if isinstance(model_cfg, dict) and "gnn_variant" in model_cfg:
                gnn_variant = str(model_cfg.get("gnn_variant", "rearev_bfs"))
            if isinstance(model_cfg, dict) and "rearev_num_instructions" in model_cfg:
                rearev_num_instructions = int(model_cfg.get("rearev_num_instructions", 3))
            if isinstance(model_cfg, dict) and "rearev_adapt_stages" in model_cfg:
                rearev_adapt_stages = int(model_cfg.get("rearev_adapt_stages", 1))

    model = RecursiveSubgraphReader(
        entity_dim=entity_dim,
        relation_dim=relation_dim,
        query_dim=query_dim,
        hidden_size=hidden_size,
        recursion_steps=recursion_steps,
        dropout=dropout,
        use_direction_embedding=bool(use_direction_embedding),
        outer_reasoning_enabled=bool(outer_reasoning_enabled),
        outer_reasoning_steps=max(1, int(outer_reasoning_steps)),
        gnn_variant=str(gnn_variant),
        rearev_num_instructions=max(1, int(rearev_num_instructions)),
        rearev_adapt_stages=max(1, int(rearev_adapt_stages)),
    )
    if ckpt_path and os.path.exists(ckpt_path):
        obj = meta if meta is not None else torch.load(ckpt_path, map_location="cpu")
        sd = obj.get("model_state", obj) if isinstance(obj, dict) else obj
        if isinstance(obj, dict):
            meta = obj
            model_cfg = obj.get("model_cfg", {})
            if isinstance(model_cfg, dict):
                if "recursion_steps" in model_cfg:
                    model.recursion_steps = max(1, int(model_cfg["recursion_steps"]))
        _safe_load_state_dict(model, sd)
    return model, meta


def train_subgraph_reader(
    args,
    *,
    is_ddp: bool,
    rank: int,
    local_rank: int,
    world_size: int,
    device: torch.device,
    wb,
):
    is_main = rank == 0
    rel2idx = load_rel_map(args.relations_txt)

    hops = int(getattr(args, "subgraph_hops", 3))
    max_nodes = int(getattr(args, "subgraph_max_nodes", 256))
    max_edges = int(getattr(args, "subgraph_max_edges", 2048))
    add_reverse_edges = _as_bool(getattr(args, "subgraph_add_reverse_edges", True))
    recursion_steps = int(getattr(args, "subgraph_recursion_steps", 8))
    dropout = float(getattr(args, "subgraph_dropout", 0.1))
    threshold = float(getattr(args, "subgraph_pred_threshold", 0.5))
    pos_weight_mode = str(getattr(args, "subgraph_pos_weight_mode", "auto")).strip().lower()
    if pos_weight_mode not in {"auto", "fixed", "off", "none", "disabled"}:
        pos_weight_mode = "auto"
    fixed_pos_weight = float(getattr(args, "subgraph_pos_weight", 1.0))
    max_pos_weight = float(getattr(args, "subgraph_pos_weight_max", 256.0))
    max_pos_weight = max(1.0, max_pos_weight)
    split_reverse_relations = _as_bool(getattr(args, "subgraph_split_reverse_relations", False))
    direction_embedding_enabled = _as_bool(
        getattr(args, "subgraph_direction_embedding_enabled", split_reverse_relations),
        default=split_reverse_relations,
    )
    outer_reasoning_enabled = _as_bool(getattr(args, "subgraph_outer_reasoning_enabled", False))
    outer_reasoning_steps = max(1, int(getattr(args, "subgraph_outer_reasoning_steps", 3)))
    gnn_variant = "rearev_bfs"
    rearev_num_instructions = max(1, int(getattr(args, "subgraph_rearev_num_ins", 3)))
    rearev_adapt_stages = max(1, int(getattr(args, "subgraph_rearev_adapt_stages", 1)))
    ranking_enabled = _as_bool(getattr(args, "subgraph_ranking_enabled", False))
    ranking_weight = max(0.0, float(getattr(args, "subgraph_ranking_weight", 0.0)))
    ranking_margin = float(getattr(args, "subgraph_ranking_margin", 0.2))
    hard_negative_topk = max(1, int(getattr(args, "subgraph_hard_negative_topk", 16)))
    bce_hard_negative_enabled = _as_bool(getattr(args, "subgraph_bce_hard_negative_enabled", False))
    bce_hard_negative_topk = max(1, int(getattr(args, "subgraph_bce_hard_negative_topk", 64)))
    lr_scheduler_mode = str(getattr(args, "subgraph_lr_scheduler", "none")).strip().lower()
    if lr_scheduler_mode not in {"none", "off", "disabled", "cosine", "step", "plateau"}:
        lr_scheduler_mode = "none"
    lr_min = max(0.0, float(getattr(args, "subgraph_lr_min", 0.0)))
    lr_step_size = max(1, int(getattr(args, "subgraph_lr_step_size", 5)))
    lr_gamma = float(getattr(args, "subgraph_lr_gamma", 0.5))
    lr_plateau_factor = float(getattr(args, "subgraph_lr_plateau_factor", 0.5))
    lr_plateau_patience = max(0, int(getattr(args, "subgraph_lr_plateau_patience", 2)))
    lr_plateau_threshold = float(getattr(args, "subgraph_lr_plateau_threshold", 1e-4))
    lr_plateau_metric = str(getattr(args, "subgraph_lr_plateau_metric", "train_loss")).strip().lower()
    if lr_plateau_metric not in {"train_loss", "dev_hit1", "dev_f1"}:
        lr_plateau_metric = "train_loss"
    grad_accum_steps = max(1, int(getattr(args, "subgraph_grad_accum_steps", 1)))
    resume_epoch_cfg = int(getattr(args, "subgraph_resume_epoch", -1))
    if resume_epoch_cfg >= 0:
        resume_epoch = int(resume_epoch_cfg)
    else:
        resume_epoch = _infer_resume_epoch_from_ckpt(getattr(args, "ckpt", ""))

    train_ds = SubgraphExampleDataset(args.train_json)
    if len(train_ds) <= 0:
        raise RuntimeError(f"Empty subgraph reader train dataset: {args.train_json}")

    train_collate = SubgraphCollator(
        entity_emb_npy=args.entity_emb_npy,
        relation_emb_npy=args.relation_emb_npy,
        query_emb_npy=args.query_emb_train_npy,
        rel2idx=rel2idx,
        hops=hops,
        max_nodes=max_nodes,
        max_edges=max_edges,
        add_reverse_edges=add_reverse_edges,
        split_reverse_relations=split_reverse_relations,
    )
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None

    # Subgraph collate uses large mmap arrays; keep workers=0 for stability.
    loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
        drop_last=False,
        collate_fn=train_collate,
        pin_memory=torch.cuda.is_available(),
    )
    if len(loader) <= 0:
        raise RuntimeError("Empty subgraph reader train loader.")

    model, _ = _load_model_from_ckpt_or_init(
        ckpt_path=getattr(args, "ckpt", ""),
        entity_dim=train_collate.entity_dim,
        relation_dim=train_collate.relation_dim,
        query_dim=train_collate.query_dim,
        hidden_size=int(args.hidden_size),
        recursion_steps=recursion_steps,
        dropout=dropout,
        use_direction_embedding=direction_embedding_enabled,
        outer_reasoning_enabled=outer_reasoning_enabled,
        outer_reasoning_steps=outer_reasoning_steps,
        gnn_variant=gnn_variant,
        rearev_num_instructions=rearev_num_instructions,
        rearev_adapt_stages=rearev_adapt_stages,
    )
    model.to(device)
    if is_ddp:
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False,
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=float(args.lr))
    scheduler = None
    if lr_scheduler_mode == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=max(1, int(args.epochs)),
            eta_min=float(lr_min),
        )
    elif lr_scheduler_mode == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=int(lr_step_size),
            gamma=float(lr_gamma),
        )
    elif lr_scheduler_mode == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max" if lr_plateau_metric in {"dev_hit1", "dev_f1"} else "min",
            factor=float(lr_plateau_factor),
            patience=int(lr_plateau_patience),
            threshold=float(lr_plateau_threshold),
        )

    has_dev = bool(getattr(args, "dev_json", "")) and os.path.exists(getattr(args, "dev_json", ""))
    dev_loader = None
    if has_dev:
        dev_ds = SubgraphExampleDataset(args.dev_json)
        eval_limit = int(getattr(args, "eval_limit", -1))
        if eval_limit > 0 and len(dev_ds) > eval_limit:
            dev_ds = Subset(dev_ds, list(range(eval_limit)))
        dev_collate = SubgraphCollator(
            entity_emb_npy=args.entity_emb_npy,
            relation_emb_npy=args.relation_emb_npy,
            query_emb_npy=getattr(args, "query_emb_dev_npy", ""),
            rel2idx=rel2idx,
            hops=hops,
            max_nodes=max_nodes,
            max_edges=max_edges,
            add_reverse_edges=add_reverse_edges,
            split_reverse_relations=split_reverse_relations,
        )
        dev_loader = DataLoader(
            dev_ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=dev_collate,
            pin_memory=torch.cuda.is_available(),
        )

    if is_main:
        print(
            "[SubgraphReader] "
            f"hops={hops} recursion_steps={recursion_steps} max_nodes={max_nodes} max_edges={max_edges} "
            f"reverse_edges={add_reverse_edges} "
            f"split_reverse_relations={split_reverse_relations} "
            f"direction_embedding={direction_embedding_enabled} "
            f"outer_reasoning={outer_reasoning_enabled} "
            f"outer_steps={outer_reasoning_steps} "
            f"gnn_variant={gnn_variant} "
            f"rearev_num_ins={rearev_num_instructions} "
            f"rearev_adapt_stages={rearev_adapt_stages} "
            f"pos_weight_mode={pos_weight_mode} "
            f"ranking_enabled={ranking_enabled} "
            f"bce_hardneg={bce_hard_negative_enabled} "
            f"lr={float(args.lr):.3e} scheduler={lr_scheduler_mode} "
            f"grad_accum={grad_accum_steps}"
        )
        if resume_epoch > 0:
            print(f"[SubgraphReader-Resume] start_from_ep={resume_epoch}")

    for local_ep in range(1, int(args.epochs) + 1):
        ep = int(resume_epoch + local_ep)
        if sampler is not None:
            sampler.set_epoch(ep)
        model.train()
        pbar = tqdm(loader, disable=not is_main, desc=f"Ep {ep} [Subgraph]")
        tot_loss = 0.0
        tot_bce_loss = 0.0
        tot_rank_loss = 0.0
        steps = 0
        pos_weight_sum = 0.0
        pos_weight_n = 0
        rank_pairs_sum = 0
        bce_keep_sum = 0
        optimizer_steps = 0
        opt.zero_grad(set_to_none=True)
        last_grad_norm = 0.0
        num_batches = len(loader)

        for batch_idx, batch in enumerate(pbar, start=1):
            batch_dev = _move_batch_to_device(batch, device)
            logits = model(
                node_emb=batch_dev["node_emb"],
                node_mask=batch_dev["node_mask"],
                seed_mask=batch_dev.get("seed_mask", None),
                edge_src=batch_dev["edge_src"],
                edge_dst=batch_dev["edge_dst"],
                edge_rel_emb=batch_dev["edge_rel_emb"],
                edge_dir=batch_dev.get("edge_dir", None),
                edge_mask=batch_dev["edge_mask"],
                q_emb=batch_dev["q_emb"],
            )
            pos_weight_t = None
            step_pos_weight = 1.0
            if pos_weight_mode == "fixed":
                step_pos_weight = max(1.0, float(fixed_pos_weight))
                pos_weight_t = torch.tensor(step_pos_weight, dtype=logits.dtype, device=logits.device)
            elif pos_weight_mode in {"off", "none", "disabled"}:
                pos_weight_t = None
                step_pos_weight = 1.0
            else:
                # auto mode: rebalance BCE by current masked positive/negative ratio.
                # This prevents easy all-negative minima when positive labels are sparse.
                with torch.no_grad():
                    m = batch_dev["node_mask"].to(torch.float32)
                    y = batch_dev["node_labels"].to(torch.float32)
                    pos = (y * m).sum()
                    tot = m.sum()
                    neg = (tot - pos).clamp(min=0.0)
                    if float(pos.item()) > 0.0:
                        step_pos_weight = float((neg / pos).item())
                    else:
                        step_pos_weight = 1.0
                step_pos_weight = float(max(1.0, min(max_pos_weight, step_pos_weight)))
                pos_weight_t = torch.tensor(step_pos_weight, dtype=logits.dtype, device=logits.device)

            bce_loss, bce_kept = _masked_bce_hard_negative_loss(
                logits,
                batch_dev["node_labels"],
                batch_dev["node_mask"],
                pos_weight=pos_weight_t,
                hard_negative_enabled=bce_hard_negative_enabled,
                hard_negative_topk=bce_hard_negative_topk,
            )
            rank_loss = logits.new_tensor(0.0)
            rank_pairs = 0
            if ranking_enabled and ranking_weight > 0.0:
                rank_loss, rank_pairs = _ranking_hard_negative_loss(
                    logits=logits,
                    targets=batch_dev["node_labels"],
                    mask=batch_dev["node_mask"],
                    margin=ranking_margin,
                    hard_negative_topk=hard_negative_topk,
                )
            loss = bce_loss + (float(ranking_weight) * rank_loss)

            if not torch.isfinite(loss):
                if is_ddp:
                    raise RuntimeError("non-finite subgraph reader loss in DDP")
                continue
            do_step = (batch_idx % grad_accum_steps == 0) or (batch_idx == num_batches)
            scaled_loss = loss / float(grad_accum_steps)
            if is_ddp and (not do_step):
                with model.no_sync():
                    scaled_loss.backward()
                grad_norm_val = float(last_grad_norm)
            else:
                scaled_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                grad_norm_val = float(grad_norm)
                last_grad_norm = grad_norm_val
                optimizer_steps += 1

            steps += 1
            tot_loss += float(loss.item())
            tot_bce_loss += float(bce_loss.item())
            tot_rank_loss += float(rank_loss.item())
            pos_weight_sum += float(step_pos_weight)
            pos_weight_n += 1
            rank_pairs_sum += int(rank_pairs)
            bce_keep_sum += int(bce_kept)

            if is_main:
                pw = pos_weight_sum / max(1, pos_weight_n)
                pbar.set_postfix_str(
                    f"loss={loss.item():.4f} bce={bce_loss.item():.4f} rank={rank_loss.item():.4f} "
                    f"avg={tot_loss/max(1,steps):.4f} pw={pw:.2f} grad={grad_norm_val:.2e}"
                )
                if wb is not None:
                    wb.log(
                        {
                            "train/step_loss": float(loss.item()),
                            "train/step_bce_loss": float(bce_loss.item()),
                            "train/step_rank_loss": float(rank_loss.item()),
                            "train/step_avg_loss": float(tot_loss / max(1, steps)),
                            "train/step_pos_weight": float(pw),
                            "train/step_rank_pairs": int(rank_pairs),
                            "train/step_bce_kept_nodes": int(bce_kept),
                            "train/grad_norm": float(grad_norm_val),
                            "train/step_optimizer_steps": int(optimizer_steps),
                            "train/epoch": int(ep),
                            "train/step": int(steps),
                        },
                        step=(ep - 1) * max(1, len(loader)) + steps,
                    )

        epoch_loss = float(tot_loss)
        epoch_bce_loss = float(tot_bce_loss)
        epoch_rank_loss = float(tot_rank_loss)
        epoch_steps = int(steps)
        epoch_rank_pairs = int(rank_pairs_sum)
        epoch_bce_kept = int(bce_keep_sum)
        if is_ddp:
            agg = torch.tensor(
                [
                    epoch_loss,
                    epoch_bce_loss,
                    epoch_rank_loss,
                    float(epoch_steps),
                    float(epoch_rank_pairs),
                    float(epoch_bce_kept),
                ],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(agg, op=dist.ReduceOp.SUM)
            epoch_loss = float(agg[0].item())
            epoch_bce_loss = float(agg[1].item())
            epoch_rank_loss = float(agg[2].item())
            epoch_steps = int(round(float(agg[3].item())))
            epoch_rank_pairs = int(round(float(agg[4].item())))
            epoch_bce_kept = int(round(float(agg[5].item())))
            dist.barrier()

        mean_loss = epoch_loss / max(1, epoch_steps)
        mean_bce = epoch_bce_loss / max(1, epoch_steps)
        mean_rank = epoch_rank_loss / max(1, epoch_steps)
        mean_rank_pairs = float(epoch_rank_pairs) / max(1, epoch_steps)
        mean_bce_kept = float(epoch_bce_kept) / max(1, epoch_steps)
        dev_hit = None
        dev_f1 = None

        if is_main:
            os.makedirs(args.out_dir, exist_ok=True)
            save_obj = model.module if hasattr(model, "module") else model
            ckpt = os.path.join(args.out_dir, f"model_ep{ep}.pt")
            payload = {
                "subgraph_reader": True,
                "epoch": int(ep),
                "model_state": save_obj.state_dict(),
                "model_cfg": {
                    "entity_dim": int(train_collate.entity_dim),
                    "relation_dim": int(train_collate.relation_dim),
                    "query_dim": int(train_collate.query_dim),
                    "hidden_size": int(save_obj.hidden_size),
                    "recursion_steps": int(save_obj.recursion_steps),
                    "dropout": float(dropout),
                    "use_direction_embedding": bool(direction_embedding_enabled),
                    "outer_reasoning_enabled": bool(outer_reasoning_enabled),
                    "outer_reasoning_steps": int(outer_reasoning_steps),
                    "gnn_variant": str(gnn_variant),
                    "rearev_num_instructions": int(rearev_num_instructions),
                    "rearev_adapt_stages": int(rearev_adapt_stages),
                },
                "subgraph_cfg": {
                    "hops": int(hops),
                    "max_nodes": int(max_nodes),
                    "max_edges": int(max_edges),
                    "add_reverse_edges": bool(add_reverse_edges),
                    "split_reverse_relations": bool(split_reverse_relations),
                    "pred_threshold": float(threshold),
                    "outer_reasoning_enabled": bool(outer_reasoning_enabled),
                    "outer_reasoning_steps": int(outer_reasoning_steps),
                    "gnn_variant": str(gnn_variant),
                    "rearev_num_instructions": int(rearev_num_instructions),
                    "rearev_adapt_stages": int(rearev_adapt_stages),
                    "grad_accum_steps": int(grad_accum_steps),
                },
            }
            torch.save(payload, ckpt)
            print(f"Saved {ckpt}")

            print(
                f"[Train-Subgraph] ep={ep} loss={mean_loss:.4f} "
                f"bce={mean_bce:.4f} rank={mean_rank:.4f} "
                f"rank_pairs/step={mean_rank_pairs:.2f} bce_kept/step={mean_bce_kept:.1f} "
                f"opt_steps={optimizer_steps}"
            )
            if wb is not None:
                wb.log(
                    {
                        "train/epoch_avg_loss": float(mean_loss),
                        "train/epoch_avg_bce_loss": float(mean_bce),
                        "train/epoch_avg_rank_loss": float(mean_rank),
                        "train/epoch_avg_rank_pairs": float(mean_rank_pairs),
                        "train/epoch_avg_bce_kept_nodes": float(mean_bce_kept),
                        "train/epoch_optimizer_steps": int(optimizer_steps),
                        "train/epoch": int(ep),
                    },
                    step=ep * max(1, len(loader)),
                )

            eval_every = max(1, int(getattr(args, "eval_every_epochs", 1)))
            eval_start = max(1, int(getattr(args, "eval_start_epoch", 1)))
            should_eval = bool(dev_loader is not None) and ep >= eval_start and ((ep - eval_start) % eval_every == 0)
            if should_eval:
                dev_hit, dev_f1, dev_skip = evaluate_subgraph_reader(
                    model=save_obj,
                    loader=dev_loader,
                    device=device,
                    pred_topk=max(1, int(getattr(args, "eval_pred_topk", 5))),
                    threshold=threshold,
                    is_main=True,
                    desc=f"Dev ep{ep} [Subgraph]",
                )
                print(f"[Dev-Subgraph] Hit@1={dev_hit:.4f} F1={dev_f1:.4f} Skip={dev_skip}")
                if wb is not None:
                    wb.log(
                        {
                            "dev/hit1": float(dev_hit),
                            "dev/f1": float(dev_f1),
                            "dev/skip": int(dev_skip),
                            "train/epoch": int(ep),
                        },
                        step=ep * max(1, len(loader)),
                )
            elif bool(getattr(args, "dev_json", "")):
                print(f"[Dev-Subgraph] skip eval at ep{ep} (start={eval_start}, every={eval_every})")

        if scheduler is not None:
            if lr_scheduler_mode == "plateau":
                if lr_plateau_metric == "dev_hit1":
                    sched_metric = float(dev_hit) if dev_hit is not None else float(-mean_loss)
                elif lr_plateau_metric == "dev_f1":
                    sched_metric = float(dev_f1) if dev_f1 is not None else float(-mean_loss)
                else:
                    sched_metric = float(mean_loss)

                if is_ddp and lr_plateau_metric in {"dev_hit1", "dev_f1"}:
                    metric_buf = torch.tensor(
                        [float(sched_metric) if is_main else 0.0],
                        dtype=torch.float64,
                        device=device,
                    )
                    dist.broadcast(metric_buf, src=0)
                    sched_metric = float(metric_buf.item())
                scheduler.step(float(sched_metric))
            else:
                scheduler.step()

            current_lr = float(opt.param_groups[0]["lr"])
            if is_main:
                print(f"[LR-Subgraph] ep={ep} lr={current_lr:.6g}")
                if wb is not None:
                    wb.log(
                        {
                            "train/lr": float(current_lr),
                            "train/epoch": int(ep),
                        },
                        step=ep * max(1, len(loader)),
                    )

        if is_ddp:
            dist.barrier()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def test_subgraph_reader(args):
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_ddp = world_size > 1
    is_main = rank == 0

    if is_ddp and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        if is_ddp:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    rel2idx = load_rel_map(args.relations_txt)
    if not getattr(args, "ckpt", ""):
        raise RuntimeError("subgraph reader test requires --ckpt")
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"checkpoint not found: {args.ckpt}")

    ckpt_obj = torch.load(args.ckpt, map_location="cpu")
    model_cfg = ckpt_obj.get("model_cfg", {}) if isinstance(ckpt_obj, dict) else {}
    sub_cfg = ckpt_obj.get("subgraph_cfg", {}) if isinstance(ckpt_obj, dict) else {}

    ent_dim = int(model_cfg.get("entity_dim", np.load(args.entity_emb_npy, mmap_mode="r").shape[1]))
    rel_dim = int(model_cfg.get("relation_dim", np.load(args.relation_emb_npy, mmap_mode="r").shape[1]))
    q_dim = int(model_cfg.get("query_dim", np.load(args.query_emb_eval_npy, mmap_mode="r").shape[1]))
    hidden_size = int(model_cfg.get("hidden_size", getattr(args, "hidden_size", 512)))
    recursion_steps = int(model_cfg.get("recursion_steps", getattr(args, "subgraph_recursion_steps", 8)))
    dropout = float(model_cfg.get("dropout", getattr(args, "subgraph_dropout", 0.1)))
    direction_embedding_enabled = _as_bool(
        getattr(args, "subgraph_direction_embedding_enabled", model_cfg.get("use_direction_embedding", False))
    )
    outer_reasoning_enabled = _as_bool(
        getattr(args, "subgraph_outer_reasoning_enabled", model_cfg.get("outer_reasoning_enabled", False))
    )
    outer_reasoning_steps = max(
        1, int(getattr(args, "subgraph_outer_reasoning_steps", model_cfg.get("outer_reasoning_steps", 3)))
    )
    gnn_variant = "rearev_bfs"
    rearev_num_instructions = max(
        1, int(getattr(args, "subgraph_rearev_num_ins", model_cfg.get("rearev_num_instructions", 3)))
    )
    rearev_adapt_stages = max(
        1, int(getattr(args, "subgraph_rearev_adapt_stages", model_cfg.get("rearev_adapt_stages", 1)))
    )

    model = RecursiveSubgraphReader(
        entity_dim=ent_dim,
        relation_dim=rel_dim,
        query_dim=q_dim,
        hidden_size=hidden_size,
        recursion_steps=recursion_steps,
        dropout=dropout,
        use_direction_embedding=direction_embedding_enabled,
        outer_reasoning_enabled=outer_reasoning_enabled,
        outer_reasoning_steps=outer_reasoning_steps,
        gnn_variant=gnn_variant,
        rearev_num_instructions=rearev_num_instructions,
        rearev_adapt_stages=rearev_adapt_stages,
    )
    sd = ckpt_obj.get("model_state", ckpt_obj) if isinstance(ckpt_obj, dict) else ckpt_obj
    skipped, missing = _safe_load_state_dict(model, sd)
    if skipped > 0:
        print(f"[warn] subgraph reader checkpoint shape-mismatch keys skipped: {skipped}")
    if missing > 0:
        print(f"[warn] subgraph reader checkpoint missing keys after load: {missing}")
    model.to(device)

    hops = int(getattr(args, "subgraph_hops", sub_cfg.get("hops", 3)))
    max_nodes = int(getattr(args, "subgraph_max_nodes", sub_cfg.get("max_nodes", 256)))
    max_edges = int(getattr(args, "subgraph_max_edges", sub_cfg.get("max_edges", 2048)))
    add_reverse_edges = _as_bool(getattr(args, "subgraph_add_reverse_edges", sub_cfg.get("add_reverse_edges", True)))
    split_reverse_relations = _as_bool(
        getattr(args, "subgraph_split_reverse_relations", sub_cfg.get("split_reverse_relations", False))
    )
    threshold = float(getattr(args, "subgraph_pred_threshold", sub_cfg.get("pred_threshold", 0.5)))

    eval_ds = SubgraphExampleDataset(args.eval_json)
    eval_limit = int(getattr(args, "eval_limit", -1))
    if eval_limit > 0 and len(eval_ds) > eval_limit:
        eval_ds = Subset(eval_ds, list(range(eval_limit)))
    eval_collate = SubgraphCollator(
        entity_emb_npy=args.entity_emb_npy,
        relation_emb_npy=args.relation_emb_npy,
        query_emb_npy=getattr(args, "query_emb_eval_npy", ""),
        rel2idx=rel2idx,
        hops=hops,
        max_nodes=max_nodes,
        max_edges=max_edges,
        add_reverse_edges=add_reverse_edges,
        split_reverse_relations=split_reverse_relations,
    )
    eval_sampler = (
        DistributedSampler(eval_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        if is_ddp
        else None
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=int(getattr(args, "batch_size", 8)),
        shuffle=False if eval_sampler is not None else False,
        sampler=eval_sampler,
        num_workers=0,
        drop_last=False,
        collate_fn=eval_collate,
        pin_memory=torch.cuda.is_available(),
    )

    if is_main:
        print("✅ subgraph-reader checkpoint loaded:", args.ckpt)
    out = evaluate_subgraph_reader(
        model=model,
        loader=eval_loader,
        device=device,
        pred_topk=max(1, int(getattr(args, "eval_pred_topk", 5))),
        threshold=threshold,
        is_main=is_main,
        desc="Test-Subgraph",
        return_counts=True,
    )
    hit, f1, skip, n_valid = out

    if is_ddp:
        # Aggregate sums across ranks for exact global metrics.
        agg = torch.tensor(
            [float(hit) * float(n_valid), float(f1) * float(n_valid), float(skip), float(n_valid)],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(agg, op=dist.ReduceOp.SUM)
        total_valid = int(round(float(agg[3].item())))
        total_skip = int(round(float(agg[2].item())))
        hit = float(agg[0].item() / max(1, total_valid))
        f1 = float(agg[1].item() / max(1, total_valid))
        skip = total_skip

    if is_main:
        print(f"[Test-Subgraph] Hit@1={hit:.4f} F1={f1:.4f} Skip={skip}")

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()
