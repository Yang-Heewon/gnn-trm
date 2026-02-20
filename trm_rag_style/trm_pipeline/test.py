from types import SimpleNamespace

from trm_unified.train_core import test as run_test


def _as_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def run(cfg):
    args = SimpleNamespace(
        trm_root=cfg['trm_root'],
        model_impl=cfg['model_impl'],
        ckpt=cfg['ckpt'],
        entities_txt=cfg['entities_txt'],
        entity_names_json=cfg.get('entity_names_json', ''),
        relations_txt=cfg['relations_txt'],
        entity_emb_npy=cfg['entity_emb_npy'],
        relation_emb_npy=cfg['relation_emb_npy'],
        eval_json=cfg.get('eval_json', ''),
        query_emb_eval_npy=cfg.get('query_emb_eval_npy', ''),
        trm_tokenizer=cfg['trm_tokenizer'],
        batch_size=int(cfg['batch_size']),
        max_steps=int(cfg.get('max_steps', 4)),
        max_q_len=int(cfg.get('max_q_len', 512)),
        max_neighbors=int(cfg.get('max_neighbors', 256)),
        prune_keep=int(cfg.get('prune_keep', 64)),
        beam=int(cfg.get('beam', 5)),
        start_topk=int(cfg.get('start_topk', 5)),
        eval_max_steps=int(cfg.get('eval_max_steps', cfg.get('max_steps', 4))),
        eval_max_neighbors=int(cfg.get('eval_max_neighbors', cfg.get('max_neighbors', 256))),
        eval_prune_keep=int(cfg.get('eval_prune_keep', cfg.get('prune_keep', 64))),
        eval_beam=int(cfg.get('eval_beam', cfg.get('beam', 5))),
        eval_start_topk=int(cfg.get('eval_start_topk', cfg.get('start_topk', 5))),
        eval_no_cycle=_as_bool(cfg.get('eval_no_cycle', False)),
        eval_limit=int(cfg.get('eval_limit', -1)),
        debug_eval_n=int(cfg.get('debug_eval_n', 5)),
        eval_pred_topk=int(cfg.get('eval_pred_topk', 1)),
        eval_use_halt=_as_bool(cfg.get('eval_use_halt', False)),
        eval_min_hops_before_stop=int(cfg.get('eval_min_hops_before_stop', 1)),
        query_residual_enabled=_as_bool(cfg.get('query_residual_enabled', False)),
        query_residual_alpha=float(cfg.get('query_residual_alpha', 0.0)),
        query_residual_mode=str(cfg.get('query_residual_mode', 'sub_rel')),
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
    )
    run_test(args)
