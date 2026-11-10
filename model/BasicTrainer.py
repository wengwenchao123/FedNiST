import torch
import math
import os
import time
import copy
import numpy as np
# import pynvml
from lib.logger import get_logger
from lib.metrics import All_Metrics
import cProfile
import random

# pynvml.nvmlInit()
# handle = pynvml.nvmlDeviceGetHandleByIndex(0)
class Trainer(object):
    def __init__(self, model, server, clients, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.server = server
        self.clients = clients
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.device = args.device
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.best_test_path = os.path.join(self.args.log_dir, 'best_test_model.pth')
        self.best_server_path = os.path.join(self.args.log_dir, 'best_server.pth')
        self.best_test_server_path = os.path.join(self.args.log_dir, 'best_test_server.pth')
        self.best_client_path = os.path.join(self.args.log_dir, 'best_client')
        self.best_test_client_path = os.path.join(self.args.log_dir, 'best_test_client')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        # log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        # if not args.debug:
        # self.logger.info("Argument: %r", args)
        # for arg, value in sorted(vars(args).items()):
        #     self.logger.info("Argument %s: %r", arg, value)

    def val_epoch(self, epoch, val_dataloader, i=2):
        self.model.eval()
        total_val_loss = 0
        epoch_time = time.time()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data
                label = target[..., :self.args.output_dim]
                output = self.model(data)
                if self.args.real_value:
                    output = self.scaler.inverse_transform(output)
                    # label = self.scaler.inverse_transform(label)
                loss = self.loss(output.cuda(), label)
                # a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('***********Val Epoch {}: average Loss: {:.6f}, train time: {:.2f} s'.format(epoch, val_loss,
                                                                                                      time.time() - epoch_time))
        return val_loss

    def test_epoch(self, epoch, test_dataloader, i=2):
        self.model.eval()
        total_test_loss = 0
        epoch_time = time.time()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_dataloader):
                data = data
                label = target[..., :self.args.output_dim]
                output = self.model(data)
                if self.args.real_value:
                    output = self.scaler.inverse_transform(output)
                    # label = self.scaler.inverse_transform(label)
                loss = self.loss(output.cuda(), label)
                # a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_test_loss += loss.item()
        test_loss = total_test_loss / len(test_dataloader)
        self.logger.info('**********test Epoch {}: average Loss: {:.6f}, train time: {:.2f} s'.format(epoch, test_loss,
                                                                                                      time.time() - epoch_time))
        return test_loss

    def train_epoch(self, epoch, i=2):
        self.model.train()
        total_loss = 0
        epoch_time = time.time()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data1 = data
            label = target[..., :self.args.output_dim]  # (..., 1)

            self.optimizer.zero_grad()

            # data and target shape: B, T, N, F; output shape: B, T, N, F
            output = self.model(data1)
            if self.args.real_value:
                # 原本这边它是只反归一化了label，修改后发现效果比论文里面的效果高了3个点，后面两年顶会能发出去真的是靠AGCRN的作者没发现这个问题
                # 上面和下面都修改了，不然就是都只反归一化label
                output = self.scaler.inverse_transform(output)
                # label = self.scaler.inverse_transform(label)

            loss = self.loss(output.cuda(), label)
            loss.backward()
            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()


            # log information
            if (batch_idx + 1) % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx + 1, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss / (self.train_per_epoch)
        self.logger.info(
            '********Train Epoch {}: averaged Loss: {:.6f}, train time: {:.2f} s'.format(epoch, train_epoch_loss,
                                                                                         time.time() - epoch_time))

        # learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        # meminfo1 = pynvml.nvmlDeviceGetMemoryInfo(handle)
        best_model = None
        best_test_model = None
        # start_time = time.time()
        not_improved_count = 0
        best_loss = float('inf')
        best_test_loss = float('inf')
        # train_loss = []
        vaild_loss = []
        test_loss = []
        train_time = []
        train_M = []
        for epoch in range(1, self.args.epochs + 1):

            # epoch_time = time.time()
            train_epoch_loss = self.train_epoch(epoch)
            # train_loss.append(train_epoch_loss)
            # self.logger.info("train time: {:.2f} s".format(time.time()-epoch_time))
            # print(time.time()-epoch_time)
            # exit()
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            test_dataloader = self.test_loader

            val_epoch_loss = self.val_epoch(epoch, val_dataloader)
            vaild_loss.append(val_epoch_loss)

            # epoch_time = time.time()
            test_epoch_loss = self.test_epoch(epoch, test_dataloader)
            # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # test_loss.append(test_epoch_loss)
            # train_time.append(time.time()-start_time)
            # train_M.append((meminfo.used-meminfo1.used)/1024**3)
            # self.logger.info("train time: {:.2f} s".format(time.time()-epoch_time))
            # print('LR:', self.optimizer.param_groups[0]['lr'])
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            # if self.val_loader == None:
            # val_epoch_loss = train_epoch_loss
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())

            if test_epoch_loss < best_test_loss:
                best_test_loss = test_epoch_loss
                best_test_model = copy.deepcopy(self.model.state_dict())

        # training_time = time.time() - start_time
        # self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))
        # np.savetxt('./{}_train_time.csv'.format(self.args.dataset), np.array([np.array(train_time),np.array(vaild_loss),np.array(test_loss),np.array(train_M)]).T,delimiter=",")

        # save the best model to file
        if not self.args.debug:
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)
            torch.save(best_test_model, self.best_test_path)
            self.logger.info("Saving current best model to " + self.best_test_path)

        # test
        self.model.load_state_dict(best_model)
        # self.val_epoch(self.args.epochs, self.test_loader)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

        self.logger.info("This is best_test_model")
        self.model.load_state_dict(best_test_model)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

    def val_epoch_fl(self, epoch, val_dataloader, i=2):
        model = self.model
        model.eval()
        total_val_loss = 0
        epoch_time = time.time()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data
                label = target[..., :self.args.output_dim]
                output = model(data)
                if self.args.real_value:
                    output = self.scaler.inverse_transform(output)
                    # label = self.scaler.inverse_transform(label)
                loss = self.loss(output.cuda(), label)
                # a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('***********Val Epoch {}: average Loss: {:.6f}, train time: {:.2f} s'.format(epoch, val_loss,
                                                                                                      time.time() - epoch_time))
        return val_loss

    def test_epoch_fl(self, epoch, test_dataloader, i=2):
        model = self.model
        model.eval()
        total_test_loss = 0
        epoch_time = time.time()
        with torch.no_grad():

            for batch_idx, (data, target) in enumerate(test_dataloader):
                data = data
                label = target[..., :self.args.output_dim]
                output = model(data)
                if self.args.real_value:
                    output = self.scaler.inverse_transform(output)
                    # label = self.scaler.inverse_transform(label)
                loss = self.loss(output.cuda(), label)
                # a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_test_loss += loss.item()
        test_loss = total_test_loss / len(test_dataloader)
        self.logger.info('**********test Epoch {}: average Loss: {:.6f}, train time: {:.2f} s'.format(epoch, test_loss,
                                                                                                      time.time() - epoch_time))
        return test_loss

    def local_val(self, client, global_weights, lr):
        # model = client.model
        # model.load_state_dict(global_weights)
        model = client.model

        model.eval()
        total_val_loss = 0
        # epoch_time = time.time()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(client.val_data_loader):
                data = data.to(self.device)
                label = target[..., :self.args.output_dim].to(self.device)
                target = target.to(self.device)
                output = model(data,target,client.client_node_id)
                if self.args.real_value:
                    output = self.scaler.inverse_transform(output)
                    # label = self.scaler.inverse_transform(label)
                loss = self.loss(output.cuda(), label)
                # a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(client.val_data_loader)
        # self.logger.info('***********Val client {}: average Loss: {:.6f}'.format(client.client_id, val_loss))
        return val_loss

    def local_test(self, client, global_weights, lr):
        # model = client.model
        # model.global_model.load_state_dict(global_weights)
        model = client.model

        model.eval()
        total_test_loss = 0
        # epoch_time = time.time()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(client.test_data_loader):
                data = data.to(self.device)
                label = target[..., :self.args.output_dim].to(self.device)
                target = target.to(self.device)
                output = model(data,target,client.client_node_id)
                if self.args.real_value:
                    output = self.scaler.inverse_transform(output)
                    # label = self.scaler.inverse_transform(label)
                loss = self.loss(output.cuda(), label)
                # a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_test_loss += loss.item()
        test_loss = total_test_loss / len(client.test_data_loader)
        # self.logger.info('***********Test client {}: average Loss: {:.6f}'.format(client.client_id, test_loss))
        return test_loss

    def local_train(self, client, global_weights, lr,epoch):
        model = client.model
        model.global_model.load_state_dict(global_weights)
        # model = self.server.global_model
        # model =self.model

        model.train()
        total_local_loss = 0
        # epoch_time = time.time()
        # optimizer = client.optimizer
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, eps=1.0e-8,
                                     weight_decay=self.args.weight_decay, amsgrad=False)
        # optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, eps=1.0e-8,
        #                              weight_decay=self.args.weight_decay, amsgrad=False)

        for i in range(1):
            total_local_loss = 0
            for batch_idx, (data, target) in enumerate(client.train_data_loader):
                data = data.to(self.device)
                label = target[..., :self.args.output_dim].to(self.device)  # (..., 1)
                target = target.to(self.device)
                optimizer.zero_grad()

                # data and target shape: B, T, N, F; output shape: B, T, N, F
                output, client_emb = model(data,target,epoch)
                if self.args.real_value:
                    # 原本这边它是只反归一化了label，修改后发现效果比论文里面的效果高了3个点，后面两年顶会能发出去真的是靠AGCRN的作者没发现这个问题
                    # 上面和下面都修改了，不然就是都只反归一化label
                    output = self.scaler.inverse_transform(output)
                    # label = self.scaler.inverse_transform(label)
                # proximal_term = 0.0
                # u = 0.01
                # for w, w_t in zip(model.global_model.parameters(), self.server.global_model.parameters()):
                #     proximal_term += (w - w_t).norm(2)

                loss = self.loss(output.cuda(), label)
                # loss = loss +(u / 2) * proximal_term
                loss.backward()

                # add max grad clipping
                if self.args.grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                optimizer.step()

                total_local_loss += loss.item()

            # #log information
            # if (batch_idx+1) % self.args.log_step == 0:
            #     self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
            #         epoch, batch_idx+1, self.train_per_epoch, loss.item()))
        train_local_loss = total_local_loss / len(client.train_data_loader)
        # self.logger.info(
        #     '********Train client {}: averaged Loss: {:.6f}'.format(client.client_id, train_local_loss))

        # #learning rate decay
        # if self.args.lr_decay:
        #     self.lr_scheduler.step()
        return model.global_model.state_dict(), train_local_loss, client_emb

    def train_fl(self):
        # meminfo1 = pynvml.nvmlDeviceGetMemoryInfo(handle)
        best_model = None
        best_test_model = None
        # start_time = time.time()
        not_improved_count = 0
        best_loss = float('inf')
        best_test_loss = float('inf')
        # train_loss = []
        vaild_loss = []

        lr = self.args.lr_init
        lr_decay_steps = []
        if getattr(self.args, 'lr_decay', False):
            lr_decay_steps = [int(i) for i in str(getattr(self.args, 'lr_decay_step', '')).split(',') if i.strip()]

        for epoch in range(1, self.args.epochs + 1):
            if epoch in lr_decay_steps:
                lr*=0.5
            epoch_time = time.time()

            global_weights = self.server.get_weights()
            client_updates = []
            loss_locals = []
            client_emb_list =[]
            # epoch_time = time.time()
            # optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr, eps=1.0e-8,
            #                              weight_decay=self.args.weight_decay, amsgrad=False)
            for client in self.clients:
                w_local, loss, client_emb = self.local_train(client, global_weights, lr,epoch)
                client_updates.append(w_local)
                loss_locals.append(loss)
                client_emb_list.append(client_emb)
            # train_epoch_loss = self.train_epoch_fl(epoch)
            # train_loss.append(train_epoch_loss)
            # self.logger.info("train time: {:.2f} s".format(time.time()-epoch_time))
            # print(time.time()-epoch_time)
            # exit()
            self.server.aggregate(client_updates,client_emb_list)

            loss_avg = sum(loss_locals) / len(loss_locals)

            self.logger.info(
                '********Train Epoch {}: averaged Loss: {:.6f}, train time: {:.2f} s'.format(epoch, loss_avg,
                                                                                             time.time() - epoch_time))

            # if self.val_loader == None:
            #     val_dataloader = self.test_loader
            # else:
            #     val_dataloader = self.val_loader
            # test_dataloader = self.test_loader
            #
            # val_epoch_loss = self.val_epoch_fl(epoch, val_dataloader)
            # vaild_loss.append(val_epoch_loss)
            #
            # # epoch_time = time.time()
            # test_epoch_loss = self.test_epoch_fl(epoch, test_dataloader)

            val_loss_locals = []
            epoch_time = time.time()
            for client in self.clients:
                loss = self.local_val(client, global_weights, lr)
                # if loss <client.client_loss:
                #     self.logger.info(
                #         '**********val client {}: average Loss: {:.6f}'.format(client.client_id, loss))
                #     torch.save(client.model.state_dict(), self.best_client_path + f'{client.client_id}.pth')
                #     self.logger.info(f"Saving current best client {client.client_id} to {self.best_client_path}")
                #     client.c
                # client_updates.append(w_local)
                val_loss_locals.append(loss)
            # train_epoch_loss = self.train_epoch_fl(epoch)
            # train_loss.append(train_epoch_loss)
            # self.logger.info("train time: {:.2f} s".format(time.time()-epoch_time))
            # print(time.time()-epoch_time)
            # exit()

            val_epoch_loss = sum(val_loss_locals) / len(val_loss_locals)
            self.logger.info(
                '**********val Epoch {}: average Loss: {:.6f}, train time: {:.2f} s'.format(epoch, val_epoch_loss,
                                                                                             time.time() - epoch_time))

            test_loss_locals = []
            epoch_time = time.time()
            for client in self.clients:
                loss = self.local_test(client, global_weights, lr)
                # if loss < client.client_test_loss:
                #     self.logger.info(
                #         '**********test client {}: average Loss: {:.6f}'.format(client.client_id, loss))
                #     torch.save(client.model.state_dict(), self.best_test_server_path + f'{client.client_id}.pth')
                #     self.logger.info(f"Saving current best client {client.client_id} to {self.best_test_server_path}")
                #     client.client_test_loss=loss
                # client_updates.append(w_local)
                test_loss_locals.append(loss)
            # train_epoch_loss = self.train_epoch_fl(epoch)
            # train_loss.append(train_epoch_loss)
            # self.logger.info("train time: {:.2f} s".format(time.time()-epoch_time))
            # print(time.time()-epoch_time)
            # exit()

            test_epoch_loss = sum(test_loss_locals) / len(test_loss_locals)
            self.logger.info(
                '**********test Epoch {}: average Loss: {:.6f}, train time: {:.2f} s'.format(epoch, test_epoch_loss,
                                                                                             time.time() - epoch_time))

            # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # test_loss.append(test_epoch_loss)
            # train_time.append(time.time()-start_time)
            # train_M.append((meminfo.used-meminfo1.used)/1024**3)
            # self.logger.info("train time: {:.2f} s".format(time.time()-epoch_time))
            # print('LR:', self.optimizer.param_groups[0]['lr'])
            if loss_avg > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            # if self.val_loader == None:
            # val_epoch_loss = train_epoch_loss
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_server_model = copy.deepcopy(self.server.global_model.state_dict())
                # best_client_model = []
                # best_client_id = []
                best_client_model=[client.model.state_dict() for client in self.clients]
                best_client_id=[client.client_id for client in self.clients]

            if test_epoch_loss < best_test_loss:
                best_test_server_model = copy.deepcopy(self.server.global_model.state_dict())
                # best_test_client_model = []
                # best_test_client_id = []
                best_test_client_model=[client.model.state_dict() for client in self.clients]
                best_test_client_id=[client.client_id for client in self.clients]

        # training_time = time.time() - start_time
        # self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))
        # np.savetxt('./{}_train_time.csv'.format(self.args.dataset), np.array([np.array(train_time),np.array(vaild_loss),np.array(test_loss),np.array(train_M)]).T,delimiter=",")
        # save the best model to file
        if not self.args.debug:
            torch.save(best_server_model, self.best_server_path)
            self.logger.info("Saving current best server to " + self.best_server_path)
            for w, i in zip(best_client_model, best_client_id):
                torch.save(w, self.best_client_path + f'{i}.pth')
                self.logger.info(f"Saving current best server {i} to {self.best_client_path} {i}")
            self.logger.info("Saving current best model to " + self.best_test_path)

            self.logger.info("Saving current best model to " + self.best_test_path)
            torch.save(best_test_server_model, self.best_test_server_path)
            for w,i in zip(best_test_client_model,best_test_client_id):
                torch.save(w, self.best_test_client_path+f'{i}.pth')
                self.logger.info(f"Saving current best test client {i} to {self.best_test_server_path} {i}")


        # # test
        self.server.global_model.load_state_dict(torch.load(self.best_server_path))
        for client in self.clients:
            client.model.load_state_dict(torch.load(self.best_client_path + f'{client.client_id}.pth'))
        # self.val_epoch(self.args.epochs, self.test_loader)
        self.test_fl(client, self.args, self.test_loader, self.scaler, self.logger)
        #
        self.server.global_model.load_state_dict(torch.load(self.best_test_server_path))
        for client in self.clients:
            client.model.load_state_dict(torch.load(self.best_test_client_path + f'{client.client_id}.pth'))
        # self.val_epoch(self.args.epochs, self.test_loader)
        self.test_fl(client, self.args, self.test_loader, self.scaler, self.logger)

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    def test_fl(self, model, args, data_loader, scaler, logger, path=None):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)

        # for client in self.clients:
        #     client.model.eval()

        # y_pred1 =[]
        # y_true1 = []
        y_pred_client =[]
        y_true_client = []
        with torch.no_grad():
            for client in self.clients:

                client.model.eval()

                y_pred = []
                y_true = []
                for batch_idx, (data, target) in enumerate(client.test_data_loader):
                    data = data.to(args.device)
                    label = target[..., :args.output_dim].to(args.device)
                    output = client.model(data,client.client_node_id)
                    y_true.append(label)
                    y_pred.append(output)

                # y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
                if args.real_value:
                    y_pred_client.append(scaler.inverse_transform(torch.cat(y_pred, dim=0)))
                    y_true_client.append(torch.cat(y_true, dim=0))
                else:
                    y_pred_client.append(torch.cat(y_pred, dim=0))
                    y_true_client.append(torch.cat(y_true, dim=0))

            # y_pred_client.append(y_pred1)
            # y_true_client.append(y_true1)
            y_pred_client = torch.cat(y_pred_client, dim=-2)
            y_true_client = torch.cat(y_true_client, dim=-2)

        # np.save('./{}_true.npy'.format(args.dataset), y_true.cpu().numpy())
        # np.save('./{}_pred.npy'.format(args.dataset), y_pred.cpu().numpy())
        for t in range(y_true_client.shape[1]):
            mae, rmse, mape, _, pcc = All_Metrics(y_pred_client[:, t, ...], y_true_client[:, t, ...],
                                                  args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}".format(
                t + 1, mae, rmse, mape))
        mae, rmse, mape, _, pcc = All_Metrics(y_pred_client, y_true_client, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}".format(
            mae, rmse, mape))

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data
                label = target[..., :args.output_dim]
                output = model(data)
                y_true.append(label)
                y_pred.append(output)

        # y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        if args.real_value:
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
            y_true = torch.cat(y_true, dim=0)
        else:
            y_pred = torch.cat(y_pred, dim=0)
            y_true = torch.cat(y_true, dim=0)
        # np.save('./{}_true.npy'.format(args.dataset), y_true.cpu().numpy())
        # np.save('./{}_pred.npy'.format(args.dataset), y_pred.cpu().numpy())
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, pcc = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                  args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}".format(
                t + 1, mae, rmse, mape))
        mae, rmse, mape, _, pcc = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}".format(
            mae, rmse, mape))
    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))
