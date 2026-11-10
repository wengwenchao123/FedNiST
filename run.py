import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import torch
import argparse
import configparser
from datetime import datetime
from model.FedNiST_arch import FedNiST,global_model
from model.BasicTrainer import Trainer
from model.Server_test import Server, Client
from lib.dataloader_FL import get_dataloader
from lib.TrainInits import print_model_parameters
import warnings
warnings.filterwarnings('ignore')


from lib.metrics import MAE_torch


def masked_mae_loss(mask_value):
    def loss(preds, labels):
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

args = argparse.ArgumentParser(description='arguments')
args.add_argument('--dataset', default='PEMSD8', type=str)
args.add_argument('--mode', default='train', type=str)
args.add_argument('--device', default='cuda:0', type=str, help='indices of GPUs')
args.add_argument('--debug', default='False', type=eval)
args.add_argument('--model', default='FedNiST', type=str)
args1, _ = args.parse_known_args()

config_file = './config_file/{}_{}.conf'.format(args1.dataset, args1.model)
config = configparser.ConfigParser()
config.read(config_file)


args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
args.add_argument('--lag', default=config['data']['lag'], type=int)
args.add_argument('--horizon', default=config['data']['horizon'], type=int)
args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
args.add_argument('--num_clients', default=config['data']['num_clients'], type=int)
args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
args.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
args.add_argument('--steps_per_day', default=config['data']['steps_per_day'], type=int)
args.add_argument('--days_per_week', default=config['data']['days_per_week'], type=int)
args.add_argument('--client_node_ratio', default='', type=str)
args.add_argument('--client_partition_method', default='random', choices=['random', 'metis_dtw'], type=str)
args.add_argument('--dtw_topk', default=8, type=int)
args.add_argument('--dtw_series_len', default=256, type=int)
args.add_argument('--dtw_window', default=24, type=int)
args.add_argument('--dtw_cache_dir', default='./data/.dtw_cache', type=str)
args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
args.add_argument('--D', default=config['model']['D'], type=int)
args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--epochs', default=config['train']['epochs'], type=int)
args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
args.add_argument('--weight_decay', default=config['train']['weight_decay'], type=float)
args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
args.add_argument('--real_value', default=config['train']['real_value'], type=eval, help = 'use real value for loss calculation')
args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
args = args.parse_args()

print(args)

if torch.cuda.is_available():
    torch.cuda.set_device(int(args.device[5]))
else:
    args.device = 'cpu'

MODEL_PARAM = {
    "num_nodes": args.num_nodes,
    "input_len": args.lag,
    "input_dim": 3,
    "out_dim": args.output_dim,
    "embed_dim": args.D,
    "output_len": args.horizon,
    "num_layer": 3,
    "if_node": True,
    "node_dim": args.D,
    "if_T_i_D": True,
    "if_D_i_W": True,
    "temp_dim_tid": 2 * args.D,
    "temp_dim_diw": args.D,
    "time_of_day_size": args.steps_per_day,
    "day_of_week_size": args.days_per_week
}

model =global_model(**MODEL_PARAM)
model = model.to(args.device)
print_model_parameters(model, only_num=False)

server =Server(model,args.device)



num_clients = args.num_clients
client_loaders, client_node_id, _, _, test_loader, scaler = get_dataloader(args,num_clients,
                                                               normalizer=args.normalizer,
                                                               dow=False,
                                                               weather=False, single=False,)
clients=[]
for i in range(num_clients):
    MODEL_PARAM['num_nodes'] = len(client_node_id[i])
    clients.append(Client(FedNiST(**MODEL_PARAM).to(args.device), i, client_loaders[i], client_node_id[i], args.lr_init,
                          args.weight_decay, args.device))

if args.loss_func == 'mask_mae':
    loss = masked_mae_loss(mask_value=0.0)
elif args.loss_func == 'mae':
    loss = torch.nn.L1Loss().to(args.device)
elif args.loss_func == 'mse':
    loss = torch.nn.MSELoss().to(args.device)
else:
    raise ValueError

current_time = datetime.now().strftime('%Y%m%d%H%M%S')
current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(current_dir,'experiments', args.dataset, current_time)
args.log_dir = log_dir

trainer = Trainer(model, server, clients, loss, None, client_loaders[0][0], client_loaders[0][1], client_loaders[0][2], scaler,
                  args)
if args.mode == 'train':
    trainer.train_fl()
elif args.mode == 'test':
    model.load_state_dict(torch.load('./pre-trained/{}.pth'.format(args.dataset)))
    print("Load saved model")
    trainer.test(model, trainer.args, test_loader, scaler, trainer.logger)
else:
    raise ValueError
