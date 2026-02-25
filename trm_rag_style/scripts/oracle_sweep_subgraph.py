#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Iterable, List
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trm_unified.data import load_rel_map
from trm_unified.subgraph_reader import (
    _build_khop_subgraph,
    _extract_gold_answers,
    _extract_seed_entities,
    _parse_tuples,
)


def _parse_int_list(raw: str) -> List[int]:
    out = []
    for x in str(raw).split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    if not out:
        raise ValueError(f"empty integer list from: {raw}")
    return out


def _parse_str_list(raw: str) -> List[str]:
    out = []
    for x in str(raw).split(","):
        x = x.strip()
        if not x:
            continue
        out.append(x)
    if not out:
        raise ValueError(f"empty string list from: {raw}")
    return out


def _as_bool(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def evaluate_oracle(
    jsonl_path: Path,
    rel2idx: dict,
    hops: int,
    max_nodes: int,
    max_edges: int,
    add_reverse_edges: bool,
    split_reverse_relations: bool,
    limit: int,
) -> dict:
    total = 0
    no_gold = 0
    any_gold = 0
    all_gold = 0
    node_sum = 0
    edge_sum = 0

    for ex in _iter_jsonl(jsonl_path):
        if limit > 0 and total >= limit:
            break
        tuples = _parse_tuples(ex.get("subgraph", {}).get("tuples", []), rel2idx)
        seeds = _extract_seed_entities(ex, tuples)
        gold = _extract_gold_answers(ex)
        if not gold:
            no_gold += 1
            continue
        total += 1
        node_cids, edges_idx = _build_khop_subgraph(
            tuples=tuples,
            seed_nodes=seeds,
            hops=hops,
            max_nodes=max_nodes,
            max_edges=max_edges,
            add_reverse_edges=add_reverse_edges,
            split_reverse_relations=split_reverse_relations,
            relation_offset=0,
        )
        node_set = set(int(x) for x in node_cids)
        gold_set = set(int(x) for x in gold)
        node_sum += int(len(node_cids))
        edge_sum += int(len(edges_idx))
        if node_set & gold_set:
            any_gold += 1
        if gold_set.issubset(node_set):
            all_gold += 1

    any_ratio = float(any_gold / total) if total > 0 else 0.0
    all_ratio = float(all_gold / total) if total > 0 else 0.0
    avg_nodes = float(node_sum / total) if total > 0 else 0.0
    avg_edges = float(edge_sum / total) if total > 0 else 0.0
    return {
        "evaluated_with_gold": total,
        "no_gold": no_gold,
        "any_gold": any_gold,
        "any_ratio": any_ratio,
        "all_gold": all_gold,
        "all_ratio": all_ratio,
        "avg_nodes": avg_nodes,
        "avg_edges": avg_edges,
    }


def main():
    ap = argparse.ArgumentParser(description="Subgraph oracle sweep for candidate ceiling.")
    ap.add_argument("--dataset", default="cwq")
    ap.add_argument("--processed-dir", default="")
    ap.add_argument("--relations-txt", default="")
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--hops-list", default="3,4")
    ap.add_argument("--max-nodes-list", default="2048,3072")
    ap.add_argument("--max-edges-list", default="8192,12288")
    ap.add_argument("--add-reverse-edges", default="true")
    ap.add_argument("--split-reverse-relations", default="false")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--out-csv", default="")
    args = ap.parse_args()

    dataset = str(args.dataset).strip().lower()
    processed_dir = Path(args.processed_dir or f"trm_agent/processed/{dataset}")
    if dataset == "cwq":
        rel_default = "data/CWQ/relations.txt"
    else:
        rel_default = "data/webqsp/relations.txt"
    relations_txt = Path(args.relations_txt or rel_default)
    out_csv = Path(args.out_csv or f"trm_rag_style/analysis/oracle_sweep_{dataset}_v2.csv")

    splits = _parse_str_list(args.splits)
    hops_list = _parse_int_list(args.hops_list)
    max_nodes_list = _parse_int_list(args.max_nodes_list)
    max_edges_list = _parse_int_list(args.max_edges_list)
    add_reverse_edges = _as_bool(args.add_reverse_edges)
    split_reverse_relations = _as_bool(args.split_reverse_relations)

    if not relations_txt.exists():
        raise FileNotFoundError(f"relations file not found: {relations_txt}")

    rel2idx = load_rel_map(str(relations_txt))
    rows = []
    for split, hops, max_nodes, max_edges in itertools.product(
        splits, hops_list, max_nodes_list, max_edges_list
    ):
        split_jsonl = processed_dir / f"{split}.jsonl"
        if not split_jsonl.exists():
            raise FileNotFoundError(f"split file not found: {split_jsonl}")
        stats = evaluate_oracle(
            jsonl_path=split_jsonl,
            rel2idx=rel2idx,
            hops=hops,
            max_nodes=max_nodes,
            max_edges=max_edges,
            add_reverse_edges=add_reverse_edges,
            split_reverse_relations=split_reverse_relations,
            limit=int(args.limit),
        )
        row = {
            "dataset": dataset,
            "split": split,
            "hops": hops,
            "max_nodes": max_nodes,
            "max_edges": max_edges,
            "add_reverse_edges": add_reverse_edges,
            "split_reverse_relations": split_reverse_relations,
            **stats,
        }
        rows.append(row)
        print(
            f"[oracle] split={split} hops={hops} nodes={max_nodes} edges={max_edges} "
            f"any={stats['any_ratio']:.4f} all={stats['all_ratio']:.4f} "
            f"avg_nodes={stats['avg_nodes']:.1f} avg_edges={stats['avg_edges']:.1f}"
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "split",
        "hops",
        "max_nodes",
        "max_edges",
        "add_reverse_edges",
        "split_reverse_relations",
        "evaluated_with_gold",
        "no_gold",
        "any_gold",
        "any_ratio",
        "all_gold",
        "all_ratio",
        "avg_nodes",
        "avg_edges",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[saved] {out_csv}")


if __name__ == "__main__":
    main()
