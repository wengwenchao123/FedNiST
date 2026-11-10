import argparse
import configparser
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
from lib.dataloader_FL import get_dataloader
from lib.metrics import MAE_torch
from model.FedNiST_arch import FedNiST, global_model
from model.server import Server
from lib.io_utils import save_json

@torch.no_grad()
def evaluate_clients(models, loaders, scaler, device, horizon):
    """Pool valid observations, rather than averaging client RMSEs."""
    total = torch.zeros(4, horizon, dtype=torch.float64, device=device)
    per_client = []

    def metrics(v):
        if v[3] <= 0:
            raise ValueError('No valid evaluation targets')
        return dict(MAE=float(v[0] / v[3]), RMSE=float((v[1] / v[3]).sqrt()), MAPE_percent=float(100 * v[2] / v[3]))
    for (model, loader) in zip(models, loaders):
        model.eval()
        sums = torch.zeros_like(total)
        for (x, y) in loader:
            truth = y[..., :1].to(device)
            pred = scaler.inverse_transform(model(x.to(device)))
            if not torch.isfinite(pred).all():
                raise FloatingPointError('Nonfinite prediction')
            mask = truth > 0
            error = (pred - truth).double()
            sums[0] += (error.abs() * mask).sum((0, 2, 3))
            sums[1] += (error.square() * mask).sum((0, 2, 3))
            sums[2] += (error.abs() / (truth + 0.001) * mask).sum((0, 2, 3))
            sums[3] += mask.sum((0, 2, 3))
        total += sums
        per_client.append(metrics(sums.sum(1)))
    return dict(overall=metrics(total.sum(1)), clients=per_client, horizons={str(i + 1): metrics(total[:, i]) for i in range(horizon)})

class BasicTrainer:
    """Train, validate, checkpoint, and test federated FedNiST."""

    def __init__(self, args):
        self.args = args

    def train(self):
        cli = self.args
        out = cli.output
        out.mkdir(parents=True, exist_ok=True)
        if (out / 'config.json').exists():
            raise FileExistsError('Experiment already exists')
        state = dict(state='preparing', completed_epochs=0)
        save_json(out / 'status.json', state)
        try:
            if cli.seed is not None:
                random.seed(cli.seed)
                np.random.seed(cli.seed)
                torch.manual_seed(cli.seed)
                torch.cuda.manual_seed_all(cli.seed)
            torch.set_num_threads(4)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            c = configparser.ConfigParser()
            source = str(cli.config_file) if cli.config_file else f'config_file/{cli.dataset}_FedNiST.conf'
            if not c.read(source):
                raise FileNotFoundError(source)
            args = SimpleNamespace(dataset=cli.dataset, temporal_split='raw-first', lazy_windows=cli.lazy_windows, client_partition_method='random', client_node_ratio='', dtw_cache_dir=cli.dtw_cache_dir, dtw_series_len=cli.dtw_series_len, dtw_window=cli.dtw_window, dtw_topk=cli.dtw_topk, partition_seed=cli.seed, column_wise=c.getboolean('data', 'column_wise'))
            for key in ('num_nodes', 'num_clients', 'lag', 'horizon', 'steps_per_day', 'days_per_week'):
                setattr(args, key, c.getint('data', key))
            if cli.num_clients is not None:
                if not 1 <= cli.num_clients <= args.num_nodes:
                    raise ValueError('num-clients must be between 1 and num_nodes')
                args.num_clients = cli.num_clients
            for key in ('val_ratio', 'test_ratio'):
                setattr(args, key, c.getfloat('data', key))
            args.input_dim = c.getint('model', 'input_dim')
            args.output_dim = c.getint('model', 'output_dim')
            args.batch_size = c.getint('train', 'batch_size')
            if args.output_dim != 1 or not c.getboolean('train', 'real_value'):
                raise ValueError('This runner requires one output feature and original-scale targets')
            d = cli.embedding_dim if cli.embedding_dim is not None else c.getint('model', 'D')
            options = dict(num_nodes=args.num_nodes, input_len=args.lag, input_dim=3, out_dim=args.output_dim, embed_dim=d, output_len=args.horizon, node_dim=d, temp_dim_tid=2 * d, temp_dim_diw=d, time_of_day_size=args.steps_per_day, day_of_week_size=args.days_per_week, if_T_i_D=True, if_D_i_W=True, if_node=True, last_block='all_moe', num_experts=cli.num_experts)
            (model_class, shared_class) = (FedNiST, global_model)
            server = Server(shared_class(**options), device=cli.device, aggregation='fedavg', projection_seed=cli.seed if cli.seed is not None else random.SystemRandom().randrange(2 ** 63))
            (groups, node_ids, _, _, _, scaler) = get_dataloader(args, args.num_clients, normalizer=c.get('data', 'normalizer'), single=False)
            models = [model_class(**dict(options, num_nodes=args.num_nodes if 'personalized' == 'all' else len(ids))).to(cli.device) for ids in node_ids]
            extra_server = None
            val_loaders = [DataLoader(g[1].dataset, batch_size=args.batch_size, shuffle=False) for g in groups]
            test_loaders = [DataLoader(g[2].dataset, batch_size=args.batch_size, shuffle=False) for g in groups]
            epochs = cli.epochs or c.getint('train', 'epochs')
            lr = c.getfloat('train', 'lr_init')
            milestones = cli.lr_decay_steps if cli.lr_decay_steps is not None else [int(s) for s in c.get('train', 'lr_decay_step').split(',')]
            if any((step < 1 for step in milestones)):
                raise ValueError('Learning-rate milestones must be positive')
            gamma = c.getfloat('train', 'lr_decay_rate', fallback=0.5)
            patience = cli.early_stop_patience if cli.early_stop_patience is not None else c.getint('train', 'early_stop_patience')
            loss_name = c.get('train', 'loss_func')
            if loss_name not in ('mae', 'mask_mae'):
                raise ValueError(loss_name)
            config = dict(vars(args), source_config=source, seed=cli.seed, last_block='all_moe', backbone='fednist', embedding_dim=d, num_experts=cli.num_experts if 'fednist' == 'fednist' and 'all_moe' == 'all_moe' else None, aggregation='equal FedAvg' if 'fedavg' == 'fedavg' else 'FedGH', local_epochs=1, epochs=epochs, lr=lr, milestones=milestones, decay_rate=gamma, decay_timing='start of round (original federated trainer convention)', early_stop=c.getboolean('train', 'early_stop'), patience=patience, loss=loss_name, optimizer='Adam reset each communication round (original federated trainer)', weight_decay=c.getfloat('train', 'weight_decay'), node_ids=node_ids, shared_keys=list(server.global_model.state_dict()), private_keys=[k for k in models[0].state_dict() if not k.startswith('global_model.')], validation='local models after local training, before next broadcast; all validation windows; pooled valid targets', test_selection='single best pooled validation MAE round', mean=float(scaler.mean), std=float(scaler.std), client_samples=[[len(loader.dataset) for loader in g] for g in groups])
            config['client_parameter_counts'] = [sum((p.numel() for p in model.parameters())) for model in models]
            config['parameter_sharing'] = 'personalized'
            config['evaluation_point'] = 'local'
            if extra_server is not None:
                config.update(private_keys=[], shared_keys=['global_model.' + k for k in server.global_model.state_dict()] + list(extra_server.values), node_aggregation='Full global node table on every client; forward selects local global IDs; equal FedAvg over all full tables including nonlocal rows', architecture=config.get('architecture', 'FedNiST four time-position concatenations').replace('private node embeddings', 'server-owned globally indexed node embeddings'))
            save_json(out / 'config.json', config)
            print(json.dumps(config), flush=True)
            (best, stale) = (float('inf'), 0)
            for epoch in range(1, epochs + 1):
                started = time.perf_counter()
                if c.getboolean('train', 'lr_decay'):
                    lr *= gamma ** milestones.count(epoch)
                shared = server.get_weights()
                losses = []
                for (client_index, (model, loaders)) in enumerate(zip(models, groups)):
                    model.global_model.load_state_dict(shared)
                    if extra_server is not None:
                        extra_server.broadcast(model, client_index)
                    model.train()
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-08, weight_decay=c.getfloat('train', 'weight_decay'))
                    loss_sum = 0.0
                    for (x, y) in loaders[0]:
                        optimizer.zero_grad(set_to_none=True)
                        (pred, _) = model(x.to(cli.device))
                        loss = MAE_torch(scaler.inverse_transform(pred), y[..., :1].to(cli.device), mask_value=0.0 if loss_name == 'mask_mae' else None)
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite training loss')
                        loss.backward()
                        if c.getboolean('train', 'grad_norm'):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), c.getfloat('train', 'max_grad_norm'))
                        optimizer.step()
                        loss_sum += loss.item()
                    losses.append(loss_sum / len(loaders[0]))
                server.aggregate([m.global_model.state_dict() for m in models])
                if extra_server is not None:
                    extra_server.aggregate(models)
                shared = server.get_weights()
                validation = evaluate_clients(models, val_loaders, scaler, cli.device, args.horizon)
                score = validation['overall']['MAE']
                if score < best:
                    (best, stale) = (score, 0)
                    torch.save(dict(epoch=epoch, server=shared, clients=[m.state_dict() for m in models], server_extra=extra_server.state_dict() if extra_server is not None else None, validation=validation, config=config), out / 'best.pt')
                else:
                    stale += 1
                record = dict(epoch=epoch, train_loss=float(np.mean(losses)), client_train_losses=losses, validation=validation, lr=lr, seconds=time.perf_counter() - started)
                if server.last_aggregation_stats is not None:
                    record['aggregation_stats'] = server.last_aggregation_stats
                    print(f'FedGH round {epoch}: ' + json.dumps(server.last_aggregation_stats), flush=True)
                with (out / 'history.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(record) + '\n')
                state.update(state='training', completed_epochs=epoch, max_epochs=epochs, last_validation_mae=score, best_validation_mae=best, lr=lr)
                save_json(out / 'status.json', state)
                print(f"Round {epoch}/{epochs} loss={record['train_loss']:.5f} val_MAE={score:.5f} best={best:.5f} lr={lr:g} seconds={record['seconds']:.1f}", flush=True)
                if config['early_stop'] and stale >= patience:
                    break
            state.update(state='evaluating')
            save_json(out / 'status.json', state)
            checkpoint = torch.load(out / 'best.pt', map_location=cli.device, weights_only=False)
            server.global_model.load_state_dict(checkpoint['server'])
            if extra_server is not None:
                extra_server.load_state_dict(checkpoint['server_extra'])
            for (model, weights) in zip(models, checkpoint['clients']):
                model.load_state_dict(weights)
            result = dict(selected_epoch=checkpoint['epoch'], validation=checkpoint['validation'], test=evaluate_clients(models, test_loaders, scaler, cli.device, args.horizon))
            save_json(out / 'metrics.json', result)
            state.update(state='complete', result=result)
            save_json(out / 'status.json', state)
            print('FINAL ' + json.dumps(result), flush=True)
        except BaseException as exc:
            state.update(state='failed', error=repr(exc))
            save_json(out / 'status.json', state)
            raise
