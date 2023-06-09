
import argparse
import os
import sys
import torch
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights
import logging
import datetime
from optimizer.amp import AMP
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torch.nn as nn
import timm
from cifar10_utils import getdata

# import import_ipynb
# import random
# import numpy as np
# import torch.backends.cudnn as cudnn
# import torchvision
# import time


class Instructor:

    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.logger.addHandler(logging.FileHandler(args.log_name))
        # self.logger.info(f"> creating model {args.model}")

        self.model = models.resnet18(pretrained=True)
        fc_in_features = self.model.fc.in_features
        num_classes = 10
        self.model.fc = torch.nn.Linear(fc_in_features, num_classes)
        rate= 0.2
        for name, module in self.model.named_children():
            if name == 'fc':
                continue
            # print(name, module)
            module = nn.Sequential(module, nn.Dropout(p=rate))
        
        # self.model = timm.create_model('resnet18', pretrained=True, num_classes=10)
        self.model.to(args.device)
        if args.device.type == 'cuda':
            self.logger.info(f"> cuda memory allocated: {torch.cuda.memory_allocated(args.device.index)}")
        self._print_args()

    # def append_dropout(self, model, rate=0.2):
    #     for name, module in self.model.named_children():
    #         if len(list(module.children())) > 0:
    #             append_dropout(module)
    #         if isinstance(module, nn.ReLU):
    #             new = nn.Sequential(module, nn.Dropout2d(p=rate, inplace=True))
    #             setattr(model, name, new)
    # append_dropout(self.model)
    
    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.size()))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        self.logger.info(f"> n_trainable_params: {n_trainable_params}, n_nontrainable_params: {n_nontrainable_params}")
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")

    def _train(self, train_dataloader, criterion, optimizer):
        train_loss, n_correct, n_train = 0, 0, 0
        n_batch = len(train_dataloader)
        self.model.train()
        for i_batch, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            def closure():
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm)
                return outputs, loss
            outputs, loss = optimizer.step(closure)
            train_loss += loss.item() * targets.size(0)
            n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
            n_train += targets.size(0)
            ratio = int((i_batch+1)*50/n_batch)
            sys.stdout.write(f"\r[{'>'*ratio}{' '*(50-ratio)}] {i_batch+1}/{n_batch} {(i_batch+1)*100/n_batch:.2f}%")
            sys.stdout.flush()
            if i_batch % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (self.args.num_epoch + 1, i_batch + 1, train_loss / 100))
                train_loss = 0.0              
        print()
        return train_loss / n_train, n_correct / n_train

    def _train2(self, train_dataloader, criterion, optimizer):
        train_loss, n_correct, n_train = 0, 0, 0
        n_batch = len(train_dataloader)
        self.model.train()
        for i_batch, (inputs, targets) in enumerate(train_dataloader, 0):
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)  # Move the input data to the GPU
            def closure():
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm)
                return outputs, loss
            outputs, loss = optimizer.step(closure)
            # train_loss += loss.item() * targets.size(0)
            train_loss += loss.item()
            n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
            n_train += targets.size(0)
            ratio = int((i_batch+1)*50/n_batch)
            sys.stdout.write(f"\r[{'>'*ratio}{' '*(50-ratio)}] {i_batch+1}/{n_batch} {(i_batch+1)*100/n_batch:.2f}%")
            sys.stdout.flush()              
            if i_batch % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (self.args.num_epoch + 1, i_batch + 1, train_loss / 100))
                train_loss = 0.0  
        print()
        return train_loss / n_train, n_correct / n_train
             


    def _test(self, test_dataloader, criterion):
        test_loss, n_correct, n_test = 0, 0, 0
        n_batch = len(test_dataloader)
        self.model.eval()
        with torch.no_grad():
            for i_batch, (inputs, targets) in enumerate(test_dataloader):
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * targets.size(0)                
                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_test += targets.size(0)
                
                ratio = int((i_batch+1)*50/n_batch)
                sys.stdout.write(f"\r[{'>'*ratio}{' '*(50-ratio)}] {i_batch+1}/{n_batch} {(i_batch+1)*100/n_batch:.2f}%")
                sys.stdout.flush()
        print()
        return test_loss / n_test, n_correct / n_test
    
    def _test2(self, test_dataloader, criterion):
        test_loss, n_correct, n_test = 0, 0, 0
        n_batch = len(test_dataloader)
        self.model.eval()
        with torch.no_grad():
            for i_batch, (inputs, targets) in enumerate(test_dataloader):
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                # print(f"{i_batch}th, targets.size(0): {targets.size(0)}")                 
                # n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                test_loss += loss.item()
                n_test += targets.size(0)
                _, predicted = torch.max(outputs.data, 1)
                n_correct += (predicted == targets).sum().item()

                ratio = int((i_batch+1)*50/n_batch)
                sys.stdout.write(f"\r[{'>'*ratio}{' '*(50-ratio)}] {i_batch+1}/{n_batch} {(i_batch+1)*100/n_batch:.2f}%")
                sys.stdout.flush()                
        print( '\nAccuracy of the network on the 10000 test images: %f %%' % (100 * n_correct / n_test))
        return test_loss / n_test, n_correct / n_test
    print('Finished Training')
 

    def run(self):
        train_dataloader, test_dataloader =  getdata(resize=self.args.resize,
                                                    batch_size=self.args.batch_size,
                                                    root_dir=os.path.join(self.args.data_dir)
                                                    # data_aug=(self.args.no_data_aug==False),
                                                    # cutout=self.args.cutout,
                                                    # autoaug=self.args.autoaug
                                                    )
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9)
        # optimizer = torch.optim.Adam(params=self.model.parameters(), lr= self.args.lr, weight_decay=self.args.weight_decay )
        # optimizer = torch.optim.AdamW(params= self.model.parameters(), lr=self.args.lr, betas=self.args.betas, eps=self.args.eps, weight_decay=self.args.weight_decay, amsgrad=False)
        # optimizer = AMP(params=filter(lambda p: p.requires_grad, self.model.parameters()),
        #                 lr=self.args.lr,
        #                 epsilon=self.args.epsilon,
        #                 inner_lr=self.args.inner_lr,
        #                 inner_iter=self.args.inner_iter,
        #                 base_optimizer=torch.optim.SGD,
        #                 momentum=self.args.momentum,
        #                 weight_decay=self.args.weight_decay,
        #                 nesterov=True)
        
        # Define the learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor= 0.1, patience= 10)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.args.milestones, self.args.gamma)
        best_loss, best_acc = 0, 0
        for epoch in range(self.args.num_epoch):
            train_loss, train_acc = self._train2(train_dataloader, criterion, optimizer)
            test_loss, test_acc = self._test2(test_dataloader, criterion)
            # scheduler.step(train_loss/100)
            scheduler.step()
            if test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss):
                best_acc, best_loss = test_acc, test_loss

            self.logger.info(f"{epoch+1}/{self.args.num_epoch} - {100*(epoch+1)/self.args.num_epoch:.2f}%")
            self.logger.info(f"[train] loss: {train_loss:.4f}, acc: {train_acc*100:.2f}, err: {100-train_acc*100:.2f}")
            self.logger.info(f"[test] loss: {test_loss:.4f}, acc: {test_acc*100:.2f}, err: {100-test_acc*100:.2f}")
        self.logger.info(f"best loss: {best_loss:.4f}, best acc: {best_acc*100:.2f}, best err: {100-best_acc*100:.2f}")
        self.logger.info(f"log saved: {self.args.log_name}")



if __name__ == "__main__":
    # model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith('__') and callable(models.__dict__[name]))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name.')
    parser.add_argument('--data_dir', type=str, default='data', help='Dictionary for dataset.')
    # parser.add_argument('--model', default='resnet18', choices=model_names, help='Model architecture.')
    parser.add_argument('--num_epoch', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=128, help= 'batch_size')
    parser.add_argument('--resize', type=int, default=256, help= 'resize')
    parser.add_argument('--gpus', type=str, default='0', help= 'epochs')
    parser.add_argument('--seed', type=int, default= '7890', help= 'seed')

    parser.add_argument('--lr', type=float, default=1e-5, help= 'lr')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help= 'weight_decay')
    # parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')

    # parser.add_argument('--betas', type=float, default=(0.9, 0.999), help='betas.')
    # parser.add_argument('--eps', type=float, default=1e-08, help='eps.')

    parser.add_argument('--clip_norm', type=int, default=50, help='Maximum norm of parameter gradient.')
    # parser.add_argument('--epsilon', type=float, default=0.5, help='Perturbation norm ball radius.')
    # parser.add_argument('--inner_lr', type=float, default=1, help='Inner learning rate.')
    # parser.add_argument('--inner_iter', type=int, default=1, help='Inner iteration number.')

    # parser.add_argument('--no_data_aug', default=False, action='store_true', help='Disable data augmentation.')
    # parser.add_argument('--cutout', default=False, action='store_true', help='Enable Cutout augmentation.')
    # parser.add_argument('--autoaug', default=False, action='store_true', help='Enable AutoAugment.')

    # parser.add_argument('--model_name', type=str, default='resnet18', help= 'model_name')
    # parser.add_argument('--ckpt', type=str, default='', help= 'ckpt')
    # parser.add_argument('--root_dir', type=str, default='drive/app/cifar10/', help= 'data_dir')

    parser.add_argument('--default_directory', type=str, default= 'drive/app/torch/save_models', help= 'default_directory')
    parser.add_argument('--optimizer', default='AdamW', help='Optimizer.')
    # parser.add_argument('--milestones', type=int, nargs='+', default=[100, 150], help='Decrease learning rate at these epochs.')
    # parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on each milstone.')

    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'], help='Device.')
    args = parser.parse_args()
    # args.num_classes = num_classes[args.dataset]
    args.log_name = f"{args.dataset}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:]}.log"
    args.device = torch.device(args.device) if args.device else torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    ins = Instructor(args)
    ins.run()

