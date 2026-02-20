#!/usr/bin/env python3
"""
Cross-platform launcher for the active paperstyle -> RL pipeline.

Works on Linux/macOS/Windows (PowerShell/CMD) as long as Python and
dependencies are available.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _bool_str(v: bool) -> str:
    return "true" if v else "false"


def _detect_cuda_device_count() -> int:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        pass
    return 0


def _default_cuda_visible_devices(cuda_count: int) -> str:
    if cuda_count <= 0:
        return ""
    use_n = min(3, int(cuda_count))
    return ",".join(str(i) for i in range(use_n))


def _default_emb_tag(model_name: str) -> str:
    raw = (model_name or "").replace("/", "__").replace(":", "__")
    cleaned = re.sub(r"[^0-9A-Za-z_.-]", "", raw)
    return cleaned or "embed"


def _print_cmd(cmd: List[str]) -> None:
    print("+ " + " ".join(shlex.quote(x) for x in cmd))


def _run(cmd: List[str], cwd: Path, env: Dict[str, str]) -> None:
    _print_cmd(cmd)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _find_latest_ckpt(ckpt_dir: Path) -> Optional[Path]:
    pat = re.compile(r"model_ep(\d+)\.pt$")
    best: Optional[Path] = None
    best_ep = -1
    if not ckpt_dir.exists():
        return None
    for p in ckpt_dir.glob("model_ep*.pt"):
        m = pat.search(p.name)
        if not m:
            continue
        ep = int(m.group(1))
        if ep > best_ep:
            best_ep = ep
            best = p
    return best


def _build_override_args(overrides: Dict[str, object]) -> List[str]:
    out = []
    for k, v in overrides.items():
        if v is None:
            continue
        if isinstance(v, bool):
            vs = _bool_str(v)
        else:
            vs = str(v)
        out.append(f"{k}={vs}")
    return out


@dataclass
class Cfg:
    repo_root: Path
    dataset: str
    model_impl: str
    emb_model: str
    emb_tag: str
    max_steps: int
    max_paths: int
    mine_max_neighbors: int
    train_path_policy: str
    train_shortest_k: int
    embed_style: str
    embed_backend: str
    embed_query_prefix: str
    embed_passage_prefix: str
    entity_names_json: str
    embed_device: str
    embed_gpus: str
    batch_size_phase1: int
    batch_size_phase2: int
    lr: float
    epochs_phase1: int
    epochs_phase2: int
    eval_limit: int
    train_style: str
    nproc_per_node: int
    nproc_per_node_phase2: int
    master_port: int
    cuda_visible_devices: str
    ddp_find_unused: bool
    ddp_timeout_minutes: int
    ddp_timeout_minutes_phase2: int
    wandb_mode: str
    wandb_project: str
    wandb_entity: str
    wandb_run_name_phase1: str
    wandb_run_name_phase2: str
    ckpt_dir_phase1: Path
    ckpt_dir_phase2: Path
    results_dir: Path
    phase1_max_neighbors: int
    phase1_prune_keep: int
    phase1_prune_rand: int
    phase1_beam: int
    phase1_start_topk: int
    phase1_debug_eval_n: int
    phase1_eval_every_epochs: int
    phase1_eval_start_epoch: int
    phase1_eval_no_cycle: bool
    phase1_eval_max_neighbors: int
    phase1_eval_prune_keep: int
    phase1_eval_beam: int
    phase1_eval_start_topk: int
    phase1_eval_pred_topk: int
    phase1_eval_use_halt: bool
    phase1_endpoint_loss_mode: str
    phase1_relation_aux_weight: float
    phase1_endpoint_aux_weight: float
    phase1_metric_align_aux_weight: float
    phase1_halt_aux_weight: float
    phase2_max_neighbors: int
    phase2_prune_keep: int
    phase2_prune_rand: int
    phase2_eval_every_epochs: int
    phase2_eval_start_epoch: int
    phase2_eval_pred_topk: int
    phase2_eval_use_halt: bool
    phase2_eval_no_cycle: bool
    phase2_warmup_endpoint_loss_mode: str
    phase2_warmup_relation_aux_weight: float
    phase2_warmup_endpoint_aux_weight: float
    phase2_warmup_metric_align_aux_weight: float
    phase2_warmup_halt_aux_weight: float
    phase2_start_epoch: int
    phase2_endpoint_loss_mode: str
    phase2_relation_aux_weight: float
    phase2_endpoint_aux_weight: float
    phase2_metric_align_aux_weight: float
    phase2_halt_aux_weight: float
    phase2_rl_reward_metric: str
    phase2_rl_entropy_weight: float
    phase2_rl_sample_temp: float
    phase2_rl_use_greedy_baseline: bool
    phase2_rl_no_cycle: bool
    phase2_rl_adv_clip: float
    train_acc_mode: str
    train_sanity_eval_every_pct: int
    train_sanity_eval_limit: int
    train_sanity_eval_beam: int
    train_sanity_eval_start_topk: int
    train_sanity_eval_pred_topk: int
    train_sanity_eval_no_cycle: bool
    train_sanity_eval_use_halt: bool
    test_eval_limit: int
    test_debug_eval_n: int
    test_batch_size: int
    test_eval_no_cycle: bool
    test_eval_max_neighbors: int
    test_eval_prune_keep: int
    test_eval_beam: int
    test_eval_start_topk: int
    test_eval_pred_topk: int
    test_eval_use_halt: bool
    test_eval_min_hops_before_stop: int
    skip_download: bool

    @staticmethod
    def from_env(repo_root: Path) -> "Cfg":
        cuda_count = _detect_cuda_device_count()
        dataset = _env("DATASET", "cwq")
        model_impl = _env("MODEL_IMPL", "trm_hier6")
        emb_model = _env("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        emb_tag = _env("EMB_TAG", _default_emb_tag(emb_model))
        epochs_phase1 = _env_int("EPOCHS_PHASE1", 5)
        epochs_phase2 = _env_int("EPOCHS_PHASE2", 20)
        ddp_timeout = _env_int("DDP_TIMEOUT_MINUTES", 180)

        wandb_name_phase1 = _env("WANDB_RUN_NAME_PHASE1", f"{dataset}_phase1_paperstyle_ep{epochs_phase1}")
        wandb_name_phase2 = _env("WANDB_RUN_NAME_PHASE2", f"{dataset}_phase2_rl_from_phase1_ep{epochs_phase2}")

        ckpt_dir_phase1 = Path(_env("CKPT_DIR_PHASE1", f"trm_rag_style/ckpt/{dataset}_{model_impl}_phase1_paperstyle"))
        ckpt_dir_phase2 = Path(_env("CKPT_DIR_PHASE2", f"trm_rag_style/ckpt/{dataset}_{model_impl}_phase2_rl_from_phase1"))
        results_dir = Path(_env("RESULTS_DIR", "experiments/paperstyle_rl/results"))

        return Cfg(
            repo_root=repo_root,
            dataset=dataset,
            model_impl=model_impl,
            emb_model=emb_model,
            emb_tag=emb_tag,
            max_steps=_env_int("MAX_STEPS", 4),
            max_paths=_env_int("MAX_PATHS", 4),
            mine_max_neighbors=_env_int("MINE_MAX_NEIGHBORS", 128),
            train_path_policy=_env("TRAIN_PATH_POLICY", "all"),
            train_shortest_k=_env_int("TRAIN_SHORTEST_K", 1),
            embed_style=_env("EMBED_STYLE", "gnn_rag_gnn_exact"),
            embed_backend=_env("EMBED_BACKEND", "sentence_transformers"),
            embed_query_prefix=_env("EMBED_QUERY_PREFIX", ""),
            embed_passage_prefix=_env("EMBED_PASSAGE_PREFIX", ""),
            entity_names_json=_env("ENTITY_NAMES_JSON", "data/data/entities_names.json"),
            embed_device=_env("EMBED_DEVICE", "cuda" if cuda_count > 0 else "cpu"),
            embed_gpus=_env("EMBED_GPUS", ""),
            batch_size_phase1=_env_int("BATCH_SIZE_PHASE1", _env_int("BATCH_SIZE", 6)),
            batch_size_phase2=_env_int("BATCH_SIZE_PHASE2", 2),
            lr=_env_float("LR", 2e-4),
            epochs_phase1=epochs_phase1,
            epochs_phase2=epochs_phase2,
            eval_limit=_env_int("EVAL_LIMIT", 200),
            train_style=_env("TRAIN_STYLE", "gnn_rag"),
            nproc_per_node=_env_int("NPROC_PER_NODE", max(1, min(3, cuda_count)) if cuda_count > 0 else 1),
            nproc_per_node_phase2=_env_int("NPROC_PER_NODE_PHASE2", 1),
            master_port=_env_int("MASTER_PORT", 29631),
            cuda_visible_devices=_env("CUDA_VISIBLE_DEVICES", _default_cuda_visible_devices(cuda_count)),
            ddp_find_unused=_env_bool("DDP_FIND_UNUSED", True),
            ddp_timeout_minutes=ddp_timeout,
            ddp_timeout_minutes_phase2=_env_int("DDP_TIMEOUT_MINUTES_PHASE2", ddp_timeout),
            wandb_mode=_env("WANDB_MODE", "online"),
            wandb_project=_env("WANDB_PROJECT", "graph-traverse"),
            wandb_entity=_env("WANDB_ENTITY", "heewon6205-chung-ang-university"),
            wandb_run_name_phase1=wandb_name_phase1,
            wandb_run_name_phase2=wandb_name_phase2,
            ckpt_dir_phase1=(repo_root / ckpt_dir_phase1),
            ckpt_dir_phase2=(repo_root / ckpt_dir_phase2),
            results_dir=(repo_root / results_dir),
            phase1_max_neighbors=_env_int("MAX_NEIGHBORS", 256),
            phase1_prune_keep=_env_int("PRUNE_KEEP", 64),
            phase1_prune_rand=_env_int("PRUNE_RAND", 64),
            phase1_beam=_env_int("BEAM", 10),
            phase1_start_topk=_env_int("START_TOPK", 5),
            phase1_debug_eval_n=_env_int("DEBUG_EVAL_N", 5),
            phase1_eval_every_epochs=_env_int("EVAL_EVERY_EPOCHS", 1),
            phase1_eval_start_epoch=_env_int("EVAL_START_EPOCH", 1),
            phase1_eval_no_cycle=_env_bool("EVAL_NO_CYCLE", False),
            phase1_eval_max_neighbors=_env_int("EVAL_MAX_NEIGHBORS", 256),
            phase1_eval_prune_keep=_env_int("EVAL_PRUNE_KEEP", 64),
            phase1_eval_beam=_env_int("EVAL_BEAM", 10),
            phase1_eval_start_topk=_env_int("EVAL_START_TOPK", 5),
            phase1_eval_pred_topk=_env_int("EVAL_PRED_TOPK", 10),
            phase1_eval_use_halt=_env_bool("EVAL_USE_HALT", False),
            phase1_endpoint_loss_mode=_env("ENDPOINT_LOSS_MODE", "aux"),
            phase1_relation_aux_weight=_env_float("RELATION_AUX_WEIGHT", 1.0),
            phase1_endpoint_aux_weight=_env_float("ENDPOINT_AUX_WEIGHT", 0.0),
            phase1_metric_align_aux_weight=_env_float("METRIC_ALIGN_AUX_WEIGHT", 0.0),
            phase1_halt_aux_weight=_env_float("HALT_AUX_WEIGHT", 0.0),
            phase2_max_neighbors=_env_int("PHASE2_MAX_NEIGHBORS", 128),
            phase2_prune_keep=_env_int("PHASE2_PRUNE_KEEP", 32),
            phase2_prune_rand=_env_int("PHASE2_PRUNE_RAND", 0),
            phase2_eval_every_epochs=_env_int("PHASE2_EVAL_EVERY_EPOCHS", 2),
            phase2_eval_start_epoch=_env_int("PHASE2_EVAL_START_EPOCH", 2),
            phase2_eval_pred_topk=_env_int("PHASE2_EVAL_PRED_TOPK", 1),
            phase2_eval_use_halt=_env_bool("PHASE2_EVAL_USE_HALT", False),
            phase2_eval_no_cycle=_env_bool("PHASE2_EVAL_NO_CYCLE", True),
            phase2_warmup_endpoint_loss_mode=_env("PHASE2_WARMUP_ENDPOINT_LOSS_MODE", _env("ENDPOINT_LOSS_MODE", "metric_align_main")),
            phase2_warmup_relation_aux_weight=_env_float("PHASE2_WARMUP_RELATION_AUX_WEIGHT", _env_float("RELATION_AUX_WEIGHT", 0.2)),
            phase2_warmup_endpoint_aux_weight=_env_float("PHASE2_WARMUP_ENDPOINT_AUX_WEIGHT", _env_float("ENDPOINT_AUX_WEIGHT", 0.05)),
            phase2_warmup_metric_align_aux_weight=_env_float("PHASE2_WARMUP_METRIC_ALIGN_AUX_WEIGHT", _env_float("METRIC_ALIGN_AUX_WEIGHT", 0.0)),
            phase2_warmup_halt_aux_weight=_env_float("PHASE2_WARMUP_HALT_AUX_WEIGHT", _env_float("HALT_AUX_WEIGHT", 0.05)),
            phase2_start_epoch=_env_int("PHASE2_START_EPOCH", 1),
            phase2_endpoint_loss_mode=_env("PHASE2_ENDPOINT_LOSS_MODE", "rl_scst"),
            phase2_relation_aux_weight=_env_float("PHASE2_RELATION_AUX_WEIGHT", 0.0),
            phase2_endpoint_aux_weight=_env_float("PHASE2_ENDPOINT_AUX_WEIGHT", 0.0),
            phase2_metric_align_aux_weight=_env_float("PHASE2_METRIC_ALIGN_AUX_WEIGHT", 0.0),
            phase2_halt_aux_weight=_env_float("PHASE2_HALT_AUX_WEIGHT", 0.0),
            phase2_rl_reward_metric=_env("PHASE2_RL_REWARD_METRIC", "f1"),
            phase2_rl_entropy_weight=_env_float("PHASE2_RL_ENTROPY_WEIGHT", 0.01),
            phase2_rl_sample_temp=_env_float("PHASE2_RL_SAMPLE_TEMP", 1.0),
            phase2_rl_use_greedy_baseline=_env_bool("PHASE2_RL_USE_GREEDY_BASELINE", False),
            phase2_rl_no_cycle=_env_bool("PHASE2_RL_NO_CYCLE", True),
            phase2_rl_adv_clip=_env_float("PHASE2_RL_ADV_CLIP", 1.0),
            train_acc_mode=_env("TRAIN_ACC_MODE", "auto"),
            train_sanity_eval_every_pct=_env_int("TRAIN_SANITY_EVAL_EVERY_PCT", 10),
            train_sanity_eval_limit=_env_int("TRAIN_SANITY_EVAL_LIMIT", 5),
            train_sanity_eval_beam=_env_int("TRAIN_SANITY_EVAL_BEAM", 5),
            train_sanity_eval_start_topk=_env_int("TRAIN_SANITY_EVAL_START_TOPK", 5),
            train_sanity_eval_pred_topk=_env_int("TRAIN_SANITY_EVAL_PRED_TOPK", 1),
            train_sanity_eval_no_cycle=_env_bool("TRAIN_SANITY_EVAL_NO_CYCLE", False),
            train_sanity_eval_use_halt=_env_bool("TRAIN_SANITY_EVAL_USE_HALT", False),
            test_eval_limit=_env_int("TEST_EVAL_LIMIT", -1),
            test_debug_eval_n=_env_int("TEST_DEBUG_EVAL_N", 5),
            test_batch_size=_env_int("TEST_BATCH_SIZE", 6),
            test_eval_no_cycle=_env_bool("TEST_EVAL_NO_CYCLE", True),
            test_eval_max_neighbors=_env_int("TEST_EVAL_MAX_NEIGHBORS", 256),
            test_eval_prune_keep=_env_int("TEST_EVAL_PRUNE_KEEP", 64),
            test_eval_beam=_env_int("TEST_EVAL_BEAM", 8),
            test_eval_start_topk=_env_int("TEST_EVAL_START_TOPK", 5),
            test_eval_pred_topk=_env_int("TEST_EVAL_PRED_TOPK", 1),
            test_eval_use_halt=_env_bool("TEST_EVAL_USE_HALT", True),
            test_eval_min_hops_before_stop=_env_int("TEST_EVAL_MIN_HOPS_BEFORE_STOP", 2),
            skip_download=_env_bool("SKIP_DOWNLOAD", False),
        )


def _trm_agent_cmd(
    stage: str,
    cfg: Cfg,
    overrides: Dict[str, object],
    *,
    ckpt: str = "",
    nproc: int = 1,
) -> List[str]:
    base = [
        "--dataset",
        cfg.dataset,
        "--model_impl",
        cfg.model_impl,
        "--stage",
        stage,
    ]
    if stage in {"embed", "train"}:
        base += ["--embedding_model", cfg.emb_model]
    if ckpt:
        base += ["--ckpt", ckpt]
    ov = _build_override_args(overrides)
    if ov:
        base += ["--override"] + ov

    if nproc > 1:
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            f"--master_port={cfg.master_port}",
            "-m",
            "trm_agent.run",
        ] + base

    return [sys.executable, "-m", "trm_agent.run"] + base


def _shared_env(cfg: Cfg) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    env["WANDB_MODE"] = cfg.wandb_mode
    env["WANDB_PROJECT"] = cfg.wandb_project
    env["WANDB_ENTITY"] = cfg.wandb_entity
    return env


def run_download(cfg: Cfg, env: Dict[str, str]) -> None:
    dl_script = cfg.repo_root / "scripts" / "download_data.sh"
    if cfg.skip_download:
        return
    if dl_script.exists():
        if shutil.which("bash"):
            _run(["bash", str(dl_script)], cwd=cfg.repo_root, env=env)
        else:
            print("[warn] bash not found. Skipping scripts/download_data.sh.")
            print("[warn] Prepare dataset files manually or rerun with --skip-download.")


def run_preprocess(cfg: Cfg, env: Dict[str, str]) -> None:

    cmd = _trm_agent_cmd(
        stage="preprocess",
        cfg=cfg,
        overrides={
            "max_steps": cfg.max_steps,
            "max_paths": cfg.max_paths,
            "mine_max_neighbors": cfg.mine_max_neighbors,
            "train_path_policy": cfg.train_path_policy,
            "train_shortest_k": cfg.train_shortest_k,
        },
    )
    _run(cmd, cwd=cfg.repo_root, env=env)


def run_embed(cfg: Cfg, env: Dict[str, str]) -> None:
    overrides: Dict[str, object] = {
        "emb_tag": cfg.emb_tag,
        "embed_device": cfg.embed_device,
        "embed_gpus": cfg.embed_gpus,
        "embed_style": cfg.embed_style,
        "embed_backend": cfg.embed_backend,
        "embed_query_prefix": cfg.embed_query_prefix,
        "embed_passage_prefix": cfg.embed_passage_prefix,
        "entity_names_json": cfg.entity_names_json,
    }
    cmd = _trm_agent_cmd(stage="embed", cfg=cfg, overrides=overrides)
    _run(cmd, cwd=cfg.repo_root, env=env)


def run_phase1(cfg: Cfg, env: Dict[str, str]) -> None:
    cfg.ckpt_dir_phase1.mkdir(parents=True, exist_ok=True)
    overrides = {
        "emb_tag": cfg.emb_tag,
        "ckpt": "",
        "ckpt_dir": str(cfg.ckpt_dir_phase1),
        "epochs": cfg.epochs_phase1,
        "batch_size": cfg.batch_size_phase1,
        "lr": cfg.lr,
        "max_steps": cfg.max_steps,
        "max_neighbors": cfg.phase1_max_neighbors,
        "prune_keep": cfg.phase1_prune_keep,
        "prune_rand": cfg.phase1_prune_rand,
        "beam": cfg.phase1_beam,
        "start_topk": cfg.phase1_start_topk,
        "train_style": cfg.train_style,
        "eval_limit": cfg.eval_limit,
        "debug_eval_n": cfg.phase1_debug_eval_n,
        "eval_every_epochs": cfg.phase1_eval_every_epochs,
        "eval_start_epoch": cfg.phase1_eval_start_epoch,
        "eval_no_cycle": cfg.phase1_eval_no_cycle,
        "eval_max_steps": cfg.max_steps,
        "eval_max_neighbors": cfg.phase1_eval_max_neighbors,
        "eval_prune_keep": cfg.phase1_eval_prune_keep,
        "eval_beam": cfg.phase1_eval_beam,
        "eval_start_topk": cfg.phase1_eval_start_topk,
        "eval_pred_topk": cfg.phase1_eval_pred_topk,
        "eval_use_halt": cfg.phase1_eval_use_halt,
        "endpoint_loss_mode": cfg.phase1_endpoint_loss_mode,
        "relation_aux_weight": cfg.phase1_relation_aux_weight,
        "endpoint_aux_weight": cfg.phase1_endpoint_aux_weight,
        "metric_align_aux_weight": cfg.phase1_metric_align_aux_weight,
        "halt_aux_weight": cfg.phase1_halt_aux_weight,
        "phase2_start_epoch": 0,
        "phase2_auto_enabled": False,
        "train_acc_mode": cfg.train_acc_mode,
        "train_sanity_eval_every_pct": cfg.train_sanity_eval_every_pct,
        "train_sanity_eval_limit": cfg.train_sanity_eval_limit,
        "train_sanity_eval_beam": cfg.train_sanity_eval_beam,
        "train_sanity_eval_start_topk": cfg.train_sanity_eval_start_topk,
        "train_sanity_eval_pred_topk": cfg.train_sanity_eval_pred_topk,
        "train_sanity_eval_no_cycle": cfg.train_sanity_eval_no_cycle,
        "train_sanity_eval_use_halt": cfg.train_sanity_eval_use_halt,
        "ddp_find_unused": cfg.ddp_find_unused,
        "wandb_mode": cfg.wandb_mode,
        "wandb_project": cfg.wandb_project,
        "wandb_entity": cfg.wandb_entity,
        "wandb_run_name": cfg.wandb_run_name_phase1,
    }
    cmd = _trm_agent_cmd(
        stage="train",
        cfg=cfg,
        overrides=overrides,
        nproc=max(1, cfg.nproc_per_node),
    )
    env_run = env.copy()
    env_run["DDP_TIMEOUT_MINUTES"] = str(cfg.ddp_timeout_minutes)
    _run(cmd, cwd=cfg.repo_root, env=env_run)


def run_phase2(cfg: Cfg, env: Dict[str, str]) -> None:
    cfg.ckpt_dir_phase2.mkdir(parents=True, exist_ok=True)
    phase1_ckpt = os.environ.get("PHASE1_CKPT", "").strip()
    ckpt_path = Path(phase1_ckpt) if phase1_ckpt else _find_latest_ckpt(cfg.ckpt_dir_phase1)
    if not ckpt_path or not ckpt_path.exists():
        raise FileNotFoundError(
            f"Phase1 checkpoint not found. Looked in {cfg.ckpt_dir_phase1}. "
            "Run phase1 first or set PHASE1_CKPT=/path/to/model_epX.pt"
        )

    overrides = {
        "emb_tag": cfg.emb_tag,
        "ckpt": str(ckpt_path),
        "ckpt_dir": str(cfg.ckpt_dir_phase2),
        "epochs": cfg.epochs_phase2,
        "batch_size": cfg.batch_size_phase2,
        "lr": cfg.lr,
        "max_steps": cfg.max_steps,
        "max_neighbors": cfg.phase2_max_neighbors,
        "prune_keep": cfg.phase2_prune_keep,
        "prune_rand": cfg.phase2_prune_rand,
        "eval_max_steps": cfg.max_steps,
        "eval_every_epochs": cfg.phase2_eval_every_epochs,
        "eval_start_epoch": cfg.phase2_eval_start_epoch,
        "eval_limit": cfg.eval_limit,
        "eval_pred_topk": cfg.phase2_eval_pred_topk,
        "eval_use_halt": cfg.phase2_eval_use_halt,
        "eval_no_cycle": cfg.phase2_eval_no_cycle,
        "endpoint_loss_mode": cfg.phase2_warmup_endpoint_loss_mode,
        "relation_aux_weight": cfg.phase2_warmup_relation_aux_weight,
        "endpoint_aux_weight": cfg.phase2_warmup_endpoint_aux_weight,
        "metric_align_aux_weight": cfg.phase2_warmup_metric_align_aux_weight,
        "halt_aux_weight": cfg.phase2_warmup_halt_aux_weight,
        "phase2_start_epoch": cfg.phase2_start_epoch,
        "phase2_endpoint_loss_mode": cfg.phase2_endpoint_loss_mode,
        "phase2_relation_aux_weight": cfg.phase2_relation_aux_weight,
        "phase2_endpoint_aux_weight": cfg.phase2_endpoint_aux_weight,
        "phase2_metric_align_aux_weight": cfg.phase2_metric_align_aux_weight,
        "phase2_halt_aux_weight": cfg.phase2_halt_aux_weight,
        "phase2_rl_reward_metric": cfg.phase2_rl_reward_metric,
        "phase2_rl_entropy_weight": cfg.phase2_rl_entropy_weight,
        "phase2_rl_sample_temp": cfg.phase2_rl_sample_temp,
        "phase2_rl_use_greedy_baseline": cfg.phase2_rl_use_greedy_baseline,
        "phase2_rl_no_cycle": cfg.phase2_rl_no_cycle,
        "phase2_rl_adv_clip": cfg.phase2_rl_adv_clip,
        "train_acc_mode": cfg.train_acc_mode,
        "train_sanity_eval_every_pct": cfg.train_sanity_eval_every_pct,
        "train_sanity_eval_limit": cfg.train_sanity_eval_limit,
        "train_sanity_eval_beam": cfg.train_sanity_eval_beam,
        "train_sanity_eval_start_topk": cfg.train_sanity_eval_start_topk,
        "train_sanity_eval_pred_topk": cfg.train_sanity_eval_pred_topk,
        "train_sanity_eval_no_cycle": cfg.train_sanity_eval_no_cycle,
        "train_sanity_eval_use_halt": cfg.train_sanity_eval_use_halt,
        "ddp_find_unused": cfg.ddp_find_unused,
        "wandb_mode": cfg.wandb_mode,
        "wandb_project": cfg.wandb_project,
        "wandb_entity": cfg.wandb_entity,
        "wandb_run_name": cfg.wandb_run_name_phase2,
    }
    cmd = _trm_agent_cmd(
        stage="train",
        cfg=cfg,
        overrides=overrides,
        nproc=max(1, cfg.nproc_per_node_phase2),
    )
    env_run = env.copy()
    env_run["DDP_TIMEOUT_MINUTES"] = str(cfg.ddp_timeout_minutes_phase2)
    env_run["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    _run(cmd, cwd=cfg.repo_root, env=env_run)


def run_test(cfg: Cfg, env: Dict[str, str]) -> None:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    phase2_ckpt = os.environ.get("PHASE2_CKPT", "").strip()
    ckpt_path = Path(phase2_ckpt) if phase2_ckpt else _find_latest_ckpt(cfg.ckpt_dir_phase2)
    if not ckpt_path or not ckpt_path.exists():
        raise FileNotFoundError(
            f"Phase2 checkpoint not found. Looked in {cfg.ckpt_dir_phase2}. "
            "Run phase2 first or set PHASE2_CKPT=/path/to/model_epX.pt"
        )
    overrides = {
        "emb_tag": cfg.emb_tag,
        "eval_limit": cfg.test_eval_limit,
        "debug_eval_n": cfg.test_debug_eval_n,
        "batch_size": cfg.test_batch_size,
        "eval_no_cycle": cfg.test_eval_no_cycle,
        "eval_max_steps": cfg.max_steps,
        "eval_max_neighbors": cfg.test_eval_max_neighbors,
        "eval_prune_keep": cfg.test_eval_prune_keep,
        "eval_beam": cfg.test_eval_beam,
        "eval_start_topk": cfg.test_eval_start_topk,
        "eval_pred_topk": cfg.test_eval_pred_topk,
        "eval_use_halt": cfg.test_eval_use_halt,
        "eval_min_hops_before_stop": cfg.test_eval_min_hops_before_stop,
    }
    cmd = _trm_agent_cmd(
        stage="test",
        cfg=cfg,
        overrides=overrides,
        ckpt=str(ckpt_path),
    )
    _run(cmd, cwd=cfg.repo_root, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-platform paperstyle pipeline runner")
    parser.add_argument(
        "--stage",
        choices=["download", "preprocess", "embed", "phase1", "phase2", "test", "all"],
        default="all",
    )
    parser.add_argument("--skip-download", action="store_true", help="skip optional download_data.sh call")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cfg = Cfg.from_env(repo_root=repo_root)
    if args.skip_download:
        cfg.skip_download = True
    env = _shared_env(cfg)

    stages = [args.stage] if args.stage != "all" else ["download", "preprocess", "embed", "phase1", "phase2", "test"]
    for st in stages:
        print(f"\n[stage] {st}")
        if st == "download":
            run_download(cfg, env)
        elif st == "preprocess":
            run_preprocess(cfg, env)
        elif st == "embed":
            run_embed(cfg, env)
        elif st == "phase1":
            run_phase1(cfg, env)
        elif st == "phase2":
            run_phase2(cfg, env)
        elif st == "test":
            run_test(cfg, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
