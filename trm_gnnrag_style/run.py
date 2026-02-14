import os

from .config_utils import build_parser, load_run_config
from .gnn import preprocess as gnn_pre
from .gnn import embed as gnn_emb
from .gnn import train as gnn_train
from .gnn import test as gnn_test


def enrich_paths(cfg):
    cfg['dataset'] = cfg['dataset'].lower()
    cfg['model_impl'] = cfg['model_impl']

    cfg.setdefault('processed_dir', os.path.join(cfg['workspace_root'], 'trm_gnnrag_style', 'processed', cfg['dataset']))
    cfg.setdefault('emb_dir', os.path.join(cfg['workspace_root'], 'trm_gnnrag_style', 'emb', f"{cfg['dataset']}_{cfg['emb_tag']}"))
    cfg.setdefault('ckpt_dir', os.path.join(cfg['workspace_root'], 'trm_gnnrag_style', 'ckpt', f"{cfg['dataset']}_{cfg['model_impl']}"))

    cfg['train_jsonl'] = os.path.join(cfg['processed_dir'], 'train.jsonl')
    cfg['dev_jsonl'] = os.path.join(cfg['processed_dir'], 'dev.jsonl')
    cfg['test_jsonl'] = os.path.join(cfg['processed_dir'], 'test.jsonl')
    cfg['train_json'] = cfg['train_jsonl']
    cfg['dev_json'] = cfg['dev_jsonl']
    cfg['eval_json'] = cfg['test_jsonl'] if os.path.exists(cfg['test_jsonl']) else cfg['dev_jsonl']

    cfg['entity_emb_npy'] = os.path.join(cfg['emb_dir'], 'entity_embeddings.npy')
    cfg['relation_emb_npy'] = os.path.join(cfg['emb_dir'], 'relation_embeddings.npy')
    cfg['query_emb_train_npy'] = os.path.join(cfg['emb_dir'], 'query_train.npy')
    cfg['query_emb_dev_npy'] = os.path.join(cfg['emb_dir'], 'query_dev.npy')
    cfg['query_emb_eval_npy'] = cfg['query_emb_dev_npy']

    return cfg


def main():
    ap = build_parser()
    args = ap.parse_args()

    cfg = load_run_config(
        config_dir=args.config_dir,
        dataset=args.dataset,
        model_impl=args.model_impl,
        embedding_model=args.embedding_model,
        ckpt=args.ckpt,
        overrides=args.override,
    )
    cfg['dataset'] = args.dataset
    cfg['model_impl'] = args.model_impl
    cfg = enrich_paths(cfg)

    stage = args.stage
    if stage in {'preprocess', 'all'}:
        gnn_pre.run(cfg)

    if stage in {'embed', 'all'}:
        gnn_emb.run(cfg)

    if stage in {'train', 'all'}:
        os.makedirs(cfg['ckpt_dir'], exist_ok=True)
        gnn_train.run(cfg)

    if stage == 'test':
        if not cfg.get('ckpt'):
            raise ValueError('stage=test requires --ckpt')
        gnn_test.run(cfg)


if __name__ == '__main__':
    main()
