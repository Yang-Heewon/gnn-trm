from types import SimpleNamespace

from trm_unified.train_core import train as run_train


def run(cfg):
    args = SimpleNamespace(
        trm_root=cfg['trm_root'],
        model_impl=cfg['model_impl'],
        train_json=cfg['train_json'],
        entities_txt=cfg['entities_txt'],
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
        eval_limit=int(cfg.get('eval_limit', -1)),
        debug_eval_n=int(cfg.get('debug_eval_n', 0)),
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
    run_train(args)
