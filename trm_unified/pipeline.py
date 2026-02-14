import argparse
import os

from .data import ensure_dir, preprocess_split
from .embedder import build_embeddings
from .train_core import train, test


def main():
    ap = argparse.ArgumentParser(description='Unified CWQ/WebQSP pipeline')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p0 = sub.add_parser('preprocess')
    p0.add_argument('--dataset', choices=['cwq', 'webqsp'], required=True)
    p0.add_argument('--train_in', required=True)
    p0.add_argument('--dev_in', required=True)
    p0.add_argument('--test_in', default='')
    p0.add_argument('--entities_txt', required=True)
    p0.add_argument('--out_dir', required=True)
    p0.add_argument('--max_steps', type=int, default=4)
    p0.add_argument('--max_paths', type=int, default=4)
    p0.add_argument('--mine_max_neighbors', type=int, default=128)

    p1 = sub.add_parser('embed')
    p1.add_argument('--model_name', default='intfloat/multilingual-e5-large')
    p1.add_argument('--entities_txt', required=True)
    p1.add_argument('--relations_txt', required=True)
    p1.add_argument('--train_jsonl', required=True)
    p1.add_argument('--dev_jsonl', required=True)
    p1.add_argument('--out_dir', required=True)
    p1.add_argument('--batch_size', type=int, default=64)
    p1.add_argument('--max_length', type=int, default=128)
    p1.add_argument('--device', default='cuda')

    p2 = sub.add_parser('train')
    p2.add_argument('--trm_root', default='/data2/workspace/heewon/논문작업/TinyRecursiveModels')
    p2.add_argument('--model_impl', choices=['trm_hier6', 'trm'], default='trm_hier6')
    p2.add_argument('--train_json', required=True)
    p2.add_argument('--entities_txt', required=True)
    p2.add_argument('--relations_txt', required=True)
    p2.add_argument('--entity_emb_npy', required=True)
    p2.add_argument('--relation_emb_npy', required=True)
    p2.add_argument('--query_emb_train_npy', default='')
    p2.add_argument('--query_emb_dev_npy', default='')
    p2.add_argument('--dev_json', default='')
    p2.add_argument('--ckpt', default='')
    p2.add_argument('--out_dir', required=True)

    p2.add_argument('--trm_tokenizer', default='intfloat/multilingual-e5-large')
    p2.add_argument('--batch_size', type=int, default=8)
    p2.add_argument('--epochs', type=int, default=5)
    p2.add_argument('--lr', type=float, default=1e-4)
    p2.add_argument('--num_workers', type=int, default=4)

    p2.add_argument('--max_steps', type=int, default=4)
    p2.add_argument('--max_q_len', type=int, default=512)
    p2.add_argument('--max_neighbors', type=int, default=256)
    p2.add_argument('--prune_keep', type=int, default=64)
    p2.add_argument('--prune_rand', type=int, default=64)
    p2.add_argument('--beam', type=int, default=5)
    p2.add_argument('--start_topk', type=int, default=5)
    p2.add_argument('--eval_limit', type=int, default=-1)
    p2.add_argument('--debug_eval_n', type=int, default=0)

    p2.add_argument('--seq_len', type=int, default=1024)
    p2.add_argument('--hidden_size', type=int, default=512)
    p2.add_argument('--num_heads', type=int, default=8)
    p2.add_argument('--expansion', type=float, default=4.0)
    p2.add_argument('--H_cycles', type=int, default=3)
    p2.add_argument('--L_cycles', type=int, default=6)
    p2.add_argument('--L_layers', type=int, default=2)
    p2.add_argument('--puzzle_emb_len', type=int, default=16)
    p2.add_argument('--pos_encodings', default='rope')
    p2.add_argument('--forward_dtype', default='float32')
    p2.add_argument('--halt_max_steps', type=int, default=16)
    p2.add_argument('--halt_exploration_prob', type=float, default=0.1)

    p3 = sub.add_parser('test')
    p3.add_argument('--trm_root', default='/data2/workspace/heewon/논문작업/TinyRecursiveModels')
    p3.add_argument('--model_impl', choices=['trm_hier6', 'trm'], default='trm_hier6')
    p3.add_argument('--ckpt', required=True)
    p3.add_argument('--entities_txt', required=True)
    p3.add_argument('--relations_txt', required=True)
    p3.add_argument('--entity_emb_npy', required=True)
    p3.add_argument('--relation_emb_npy', required=True)
    p3.add_argument('--eval_json', default='')
    p3.add_argument('--query_emb_eval_npy', default='')
    p3.add_argument('--trm_tokenizer', default='intfloat/multilingual-e5-large')
    p3.add_argument('--batch_size', type=int, default=8)
    p3.add_argument('--max_steps', type=int, default=4)
    p3.add_argument('--max_q_len', type=int, default=512)
    p3.add_argument('--max_neighbors', type=int, default=256)
    p3.add_argument('--prune_keep', type=int, default=64)
    p3.add_argument('--beam', type=int, default=5)
    p3.add_argument('--start_topk', type=int, default=5)
    p3.add_argument('--eval_limit', type=int, default=-1)
    p3.add_argument('--debug_eval_n', type=int, default=5)
    p3.add_argument('--seq_len', type=int, default=1024)
    p3.add_argument('--hidden_size', type=int, default=512)
    p3.add_argument('--num_heads', type=int, default=8)
    p3.add_argument('--expansion', type=float, default=4.0)
    p3.add_argument('--H_cycles', type=int, default=3)
    p3.add_argument('--L_cycles', type=int, default=6)
    p3.add_argument('--L_layers', type=int, default=2)
    p3.add_argument('--puzzle_emb_len', type=int, default=16)
    p3.add_argument('--pos_encodings', default='rope')
    p3.add_argument('--forward_dtype', default='float32')
    p3.add_argument('--halt_max_steps', type=int, default=16)
    p3.add_argument('--halt_exploration_prob', type=float, default=0.1)

    args = ap.parse_args()

    if args.cmd == 'preprocess':
        ensure_dir(args.out_dir)
        tr = preprocess_split(args.dataset, args.train_in, os.path.join(args.out_dir, 'train.jsonl'), args.entities_txt, args.max_steps, args.max_paths, args.mine_max_neighbors)
        dv = preprocess_split(args.dataset, args.dev_in, os.path.join(args.out_dir, 'dev.jsonl'), args.entities_txt, args.max_steps, args.max_paths, args.mine_max_neighbors)
        print('train:', tr)
        print('dev:', dv)
        if args.test_in:
            te = preprocess_split(args.dataset, args.test_in, os.path.join(args.out_dir, 'test.jsonl'), args.entities_txt, args.max_steps, args.max_paths, args.mine_max_neighbors)
            print('test:', te)

    elif args.cmd == 'embed':
        meta = build_embeddings(args.model_name, args.entities_txt, args.relations_txt, args.train_jsonl, args.dev_jsonl, args.out_dir, args.batch_size, args.max_length, args.device)
        print(meta)

    elif args.cmd == 'train':
        train(args)

    elif args.cmd == 'test':
        test(args)


if __name__ == '__main__':
    main()
