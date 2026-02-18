import os

from .config_utils import build_parser, load_run_config
from .trm_pipeline import preprocess as trm_pre
from .trm_pipeline import embed as trm_emb
from .trm_pipeline import train as trm_train
from .trm_pipeline import test as trm_test


def _resolve_path(base_dir, raw):
    if not isinstance(raw, str) or not raw:
        return raw
    p = os.path.expanduser(os.path.expandvars(raw))
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(base_dir, p))


def normalize_config_paths(cfg):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    workspace_root = cfg.get('workspace_root') or repo_root
    workspace_root = _resolve_path(repo_root, workspace_root)
    cfg['workspace_root'] = workspace_root

    if not cfg.get('trm_root'):
        cfg['trm_root'] = os.environ.get('TRM_ROOT', 'TinyRecursiveModels')

    path_keys = [
        'trm_root',
        'train_in',
        'dev_in',
        'test_in',
        'entities_txt',
        'entity_names_json',
        'relations_txt',
        'processed_dir',
        'emb_dir',
        'ckpt_dir',
        'train_jsonl',
        'dev_jsonl',
        'test_jsonl',
        'train_json',
        'dev_json',
        'eval_json',
        'entity_emb_npy',
        'relation_emb_npy',
        'query_emb_train_npy',
        'query_emb_dev_npy',
        'query_emb_eval_npy',
        'ckpt',
    ]
    for k in path_keys:
        if k in cfg:
            cfg[k] = _resolve_path(workspace_root, cfg[k])
    return cfg


def enrich_paths(cfg):
    cfg['dataset'] = cfg['dataset'].lower()
    cfg['model_impl'] = cfg['model_impl']
    had_processed_dir = 'processed_dir' in cfg
    had_emb_dir = 'emb_dir' in cfg
    had_ckpt_dir = 'ckpt_dir' in cfg

    cfg.setdefault('processed_dir', os.path.join(cfg['workspace_root'], 'trm_agent', 'processed', cfg['dataset']))
    cfg.setdefault('emb_dir', os.path.join(cfg['workspace_root'], 'trm_agent', 'emb', f"{cfg['dataset']}_{cfg['emb_tag']}"))
    cfg.setdefault('ckpt_dir', os.path.join(cfg['workspace_root'], 'trm_agent', 'ckpt', f"{cfg['dataset']}_{cfg['model_impl']}"))

    # Backward compatibility: if new default dirs do not exist, transparently
    # reuse legacy trm_rag_style outputs.
    if not had_processed_dir and not os.path.exists(cfg['processed_dir']):
        legacy_processed = os.path.join(cfg['workspace_root'], 'trm_rag_style', 'processed', cfg['dataset'])
        if os.path.exists(legacy_processed):
            cfg['processed_dir'] = legacy_processed
    if not had_emb_dir:
        # Prefer legacy emb dir when default dir exists but is empty/incomplete.
        default_ent = os.path.join(cfg['emb_dir'], 'entity_embeddings.npy')
        if not os.path.exists(default_ent):
            legacy_emb = os.path.join(cfg['workspace_root'], 'trm_rag_style', 'emb', f"{cfg['dataset']}_{cfg['emb_tag']}")
            legacy_ent = os.path.join(legacy_emb, 'entity_embeddings.npy')
            if os.path.exists(legacy_ent):
                cfg['emb_dir'] = legacy_emb
    if not had_ckpt_dir and not os.path.exists(cfg['ckpt_dir']):
        legacy_ckpt = os.path.join(cfg['workspace_root'], 'trm_rag_style', 'ckpt', f"{cfg['dataset']}_{cfg['model_impl']}")
        if os.path.exists(legacy_ckpt):
            cfg['ckpt_dir'] = legacy_ckpt

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
    cfg = normalize_config_paths(cfg)
    cfg = enrich_paths(cfg)
    cfg = normalize_config_paths(cfg)

    stage = args.stage
    if stage in {'preprocess', 'all'}:
        trm_pre.run(cfg)

    if stage in {'embed', 'all'}:
        trm_emb.run(cfg)

    if stage in {'train', 'all'}:
        os.makedirs(cfg['ckpt_dir'], exist_ok=True)
        trm_train.run(cfg)

    if stage == 'test':
        if not cfg.get('ckpt'):
            raise ValueError('stage=test requires --ckpt')
        trm_test.run(cfg)


if __name__ == '__main__':
    main()
