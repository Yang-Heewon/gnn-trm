from types import SimpleNamespace

from trm_unified.train_core import train as run_train


def _as_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def run(cfg):
    args = SimpleNamespace(
        trm_root=cfg['trm_root'],
        model_impl=cfg['model_impl'],
        train_json=cfg['train_json'],
        entities_txt=cfg['entities_txt'],
        entity_names_json=cfg.get('entity_names_json', ''),
        relations_txt=cfg['relations_txt'],
        entity_emb_npy=cfg['entity_emb_npy'],
        relation_emb_npy=cfg['relation_emb_npy'],
        query_emb_train_npy=cfg['query_emb_train_npy'],
        query_emb_dev_npy=cfg.get('query_emb_dev_npy', ''),
        dev_json=cfg.get('dev_json', ''),
        ckpt=cfg.get('ckpt', ''),
        out_dir=cfg['ckpt_dir'],
        trm_tokenizer=cfg['trm_tokenizer'],
        batch_size=int(cfg['batch_size']),
        epochs=int(cfg['epochs']),
        lr=float(cfg['lr']),
        num_workers=int(cfg['num_workers']),
        max_steps=int(cfg['max_steps']),
        max_q_len=int(cfg['max_q_len']),
        max_neighbors=int(cfg['max_neighbors']),
        prune_keep=int(cfg['prune_keep']),
        prune_rand=int(cfg['prune_rand']),
        beam=int(cfg.get('beam', 5)),
        start_topk=int(cfg.get('start_topk', 5)),
        eval_beam=int(cfg.get('eval_beam', cfg.get('beam', 5))),
        eval_start_topk=int(cfg.get('eval_start_topk', cfg.get('start_topk', 5))),
        eval_max_steps=int(cfg.get('eval_max_steps', cfg['max_steps'])),
        eval_max_neighbors=int(cfg.get('eval_max_neighbors', cfg['max_neighbors'])),
        eval_prune_keep=int(cfg.get('eval_prune_keep', cfg['prune_keep'])),
        eval_no_cycle=_as_bool(cfg.get('eval_no_cycle', False)),
        eval_limit=int(cfg.get('eval_limit', -1)),
        debug_eval_n=int(cfg.get('debug_eval_n', 0)),
        eval_pred_topk=int(cfg.get('eval_pred_topk', 1)),
        eval_use_halt=_as_bool(cfg.get('eval_use_halt', False)),
        eval_min_hops_before_stop=int(cfg.get('eval_min_hops_before_stop', 1)),
        eval_every_epochs=int(cfg.get('eval_every_epochs', 1)),
        eval_start_epoch=int(cfg.get('eval_start_epoch', 1)),
        endpoint_aux_weight=float(cfg.get('endpoint_aux_weight', 0.0)),
        halt_aux_weight=float(cfg.get('halt_aux_weight', 0.0)),
        oracle_diag_enabled=_as_bool(cfg.get('oracle_diag_enabled', True)),
        oracle_diag_limit=int(cfg.get('oracle_diag_limit', -1)),
        oracle_diag_fail_threshold=float(cfg.get('oracle_diag_fail_threshold', -1.0)),
        oracle_diag_only=_as_bool(cfg.get('oracle_diag_only', False)),
        seq_len=int(cfg['seq_len']),
        hidden_size=int(cfg['hidden_size']),
        num_heads=int(cfg['num_heads']),
        expansion=float(cfg['expansion']),
        H_cycles=int(cfg['H_cycles']),
        L_cycles=int(cfg['L_cycles']),
        L_layers=int(cfg['L_layers']),
        puzzle_emb_len=int(cfg['puzzle_emb_len']),
        pos_encodings=cfg['pos_encodings'],
        forward_dtype=cfg['forward_dtype'],
        halt_max_steps=int(cfg['halt_max_steps']),
        halt_exploration_prob=float(cfg['halt_exploration_prob']),
        ddp_find_unused=bool(cfg.get('ddp_find_unused', False)),
        wandb_project=cfg.get('wandb_project', ''),
        wandb_entity=cfg.get('wandb_entity', ''),
        wandb_run_name=cfg.get('wandb_run_name', ''),
        wandb_mode=cfg.get('wandb_mode', 'disabled'),
    )
    run_train(args)
