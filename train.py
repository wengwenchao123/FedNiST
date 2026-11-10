import argparse
from pathlib import Path
from model.BasicTrainer import BasicTrainer

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='PEMSD8')
    parser.add_argument('--config-file', type=Path, help='Optional frozen dataset configuration')
    parser.add_argument('--num-clients', type=int, default=None)
    parser.add_argument('--dtw-cache-dir', default='./data/.dtw_cache')
    parser.add_argument('--dtw-series-len', type=int, default=256)
    parser.add_argument('--dtw-window', type=int, default=24)
    parser.add_argument('--dtw-topk', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    seed_options = parser.add_mutually_exclusive_group()
    seed_options.add_argument('--seed', type=int, default=0)
    seed_options.add_argument('--no-seed', action='store_true')
    parser.add_argument('--early-stop-patience', type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr-decay-steps', type=int, nargs='+')
    parser.add_argument('--lazy-windows', action='store_true')
    parser.add_argument('--embedding-dim', type=int, help='Override base width D from the configuration')
    parser.add_argument('--num-experts', type=int, default=4, help='Dense experts in each FedNiST MoE layer')
    cli = parser.parse_args(argv)
    if cli.num_experts < 1 or (cli.embedding_dim is not None and cli.embedding_dim < 1):
        parser.error('embedding-dim and num-experts must be positive')
    if cli.no_seed:
        cli.seed = None
    if cli.early_stop_patience is not None and cli.early_stop_patience < 1:
        parser.error('--early-stop-patience must be positive')
    BasicTrainer(cli).train()
if __name__ == '__main__':
    main()
