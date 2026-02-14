from trm_unified.embedder import build_embeddings


def run(cfg):
    meta = build_embeddings(
        model_name=cfg['model_name'],
        entities_txt=cfg['entities_txt'],
        relations_txt=cfg['relations_txt'],
        train_jsonl=cfg['train_jsonl'],
        dev_jsonl=cfg['dev_jsonl'],
        out_dir=cfg['emb_dir'],
        batch_size=int(cfg['embed_batch_size']),
        max_length=int(cfg['embed_max_length']),
        device=cfg['embed_device'],
    )
    print('✅ embed done:', meta)
    return meta
