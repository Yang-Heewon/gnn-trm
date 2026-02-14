import os

from trm_unified.data import ensure_dir, preprocess_split


def run(cfg):
    ensure_dir(cfg['processed_dir'])

    train_out = os.path.join(cfg['processed_dir'], 'train.jsonl')
    dev_out = os.path.join(cfg['processed_dir'], 'dev.jsonl')

    tr = preprocess_split(
        dataset=cfg['dataset'],
        input_path=cfg['train_in'],
        output_path=train_out,
        entities_txt=cfg['entities_txt'],
        max_steps=int(cfg['max_steps']),
        max_paths=int(cfg['max_paths']),
        max_neighbors=int(cfg['mine_max_neighbors']),
    )
    dv = preprocess_split(
        dataset=cfg['dataset'],
        input_path=cfg['dev_in'],
        output_path=dev_out,
        entities_txt=cfg['entities_txt'],
        max_steps=int(cfg['max_steps']),
        max_paths=int(cfg['max_paths']),
        max_neighbors=int(cfg['mine_max_neighbors']),
    )

    out = {'train': tr, 'dev': dv}
    if cfg.get('test_in'):
        test_out = os.path.join(cfg['processed_dir'], 'test.jsonl')
        te = preprocess_split(
            dataset=cfg['dataset'],
            input_path=cfg['test_in'],
            output_path=test_out,
            entities_txt=cfg['entities_txt'],
            max_steps=int(cfg['max_steps']),
            max_paths=int(cfg['max_paths']),
            max_neighbors=int(cfg['mine_max_neighbors']),
        )
        out['test'] = te

    print('✅ preprocess done:', out)
    return out
