import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import copy
import time

from utils.args import parse_args

args = parse_args()

class Block():
    def __init__(self, block_id, data_x, data_y):
        self.block_id = block_id
        self.size = args.block_size
        self.data_x = data_x
        self.data_y = data_y

class Client():
    def __init__(self, client_id, train_data, test_data):
        self.client_id = client_id
        self.train_data = {}
        self.test_data = test_data

        self.block_count = 0
        self.blocks = []

        assert len(train_data['x']) % args.block_size == 0, "The block number must be integer"
        for i in range(0, len(train_data['x']), args.block_size):
            self.create_block(train_data['x'][i:i+args.block_size], train_data['y'][i:i+args.block_size])
      
    def create_block(self, data_x, data_y):
        assert len(data_x) == args.block_size, "Dataset size must equal to block size"
        self.block_count = self.block_count + 1
        self.blocks.append(Block(self.block_count, data_x, data_y))

    def select_blocks(self, blocks):
        self.train_data = {'x': [], 'y': []}
        for block_id in blocks:
            self.train_data['x'].extend(self.blocks[block_id-1].data_x)
            self.train_data['y'].extend(self.blocks[block_id-1].data_y)
    
    def train(self, task, blocks): # placement
        def flatten_gradients(gradients):
            gradients_flattened = torch.empty(0, device=args.cuda)
            gradients_shape = []
            gradients_length = []

            for gradient in gradients:
                gradients_shape.append(gradient.shape)
                gradient = gradient.flatten()
                gradients_flattened = torch.cat((gradients_flattened, gradient), dim=0)
                gradients_length.append(gradient.shape[0])
            return gradients_flattened, gradients_shape, gradients_length
        
        def reshape_gradients(gradients_noised, gradients_shape, gradients_length):
            gradients_reshaped = []
            length = 0
            for i in range(len(gradients_length)):
                temp = gradients_noised[length:length+gradients_length[i]]
                length = length + gradients_length[i]
                gradients_reshaped.append(torch.reshape(temp, gradients_shape[i]))
            return gradients_reshaped

        def private_gradients(gradients, eps0, l2_norm_clip):
            l2_norm = float(torch.linalg.norm(gradients))
            grads = gradients / max(1.0 * l2_norm / l2_norm_clip, 1.0)

            l2_norm = float(torch.linalg.norm(grads))
            pr = 0.5 + l2_norm / (2 * l2_norm_clip)
            pr = pr if pr < 1.0 else 1.0
            x = 2 * np.random.binomial(1, pr, 1)[0] - 1
            grads = grads * (x * l2_norm_clip / l2_norm)
            d = grads.shape[0]
            C = (np.exp(eps0) + 1) / (np.exp(eps0) - 1)
            M = C * l2_norm_clip * np.sqrt(np.pi * d / 2)
            V = torch.randn(d, device=args.cuda)
            V = V / float(torch.linalg.norm(V))
            if (torch.dot(V, grads) < 0):
                V = -V
            pr = (np.exp(eps0)) / (np.exp(eps0) + 1)
            x = 2 * np.random.binomial(1, pr, 1)[0] - 1
            V = V * (x * M)
            return V
        
        self.select_blocks(blocks)
        gradients, layer_name_list = self.train_epoch(task, task.trained_round, args.batch_size)

        # start_time = time.time()
        gradients_flattened, gradients_shape, gradients_length = flatten_gradients(gradients)
        l2_norm_clip = args.l2_norm_clip_LeNet if task.model_type == 'small' else args.l2_norm_clip_ResNet
        gradients_noised = private_gradients(gradients_flattened, task.eps0, l2_norm_clip)
        gradients_reshaped = reshape_gradients(gradients_noised, gradients_shape, gradients_length)
        # end_time = time.time()
        # print(f'duartion  = {end_time - start_time}')
        return len(self.train_data['y']), gradients_reshaped

    def train_epoch(self, task, epoch, batch_size):
        model = task.model.to(args.cuda)
        model.train()
        task.model.load_state_dict(task.model_before)
        old_para = task.model_before
        data_loader = self.utils_loader(self.train_data, batch_size, 'train')
        for i, (inputs, target) in enumerate(data_loader):
            inputs = inputs.to(args.cuda)
            target = target.to(args.cuda)
            output = model(inputs)
            model.criterion.to(args.cuda)
            loss = model.criterion(output, target)
            model.optimizer.update(epoch, epoch * len(data_loader) + i)
            model.optimizer.zero_grad()
            loss.backward()
            name_list = []
            grads_list = []
            for name, param in model.named_parameters():
                name_list.append(name)
                grads_list.append(param.grad)
            model.optimizer.step()

        new_para = model.state_dict()
        layer_name_list_old, layer_val_list_old = list(old_para.keys()), list(old_para.values())
        layer_name_list_new, layer_val_list_new = list(new_para.keys()), list(new_para.values())
        gradients = []
        for i in range(len(layer_val_list_old)):
            gradients.append(layer_val_list_old[i] - layer_val_list_new[i])
        return gradients, layer_name_list_new

    def test(self, task):
        model = task.model.to(args.cuda)
        model.eval()

        from meters import AverageMeter, accuracy
        losses = AverageMeter()
        top1 = AverageMeter()

        batch_size=50
        data_loader = self.utils_loader(self.test_data, batch_size, 'test')
        for i, (inputs, target) in enumerate(data_loader):
            inputs = inputs.to(args.cuda)
            target = target.to(args.cuda)
            output = model(inputs)
            loss = model.criterion(output, target)
            losses.update(float(loss), inputs.size(0))
            prec1, prec5 = accuracy(output.detach(), target, topk=(1, 5))
            top1.update(float(prec1), inputs.size(0))
        
        return top1.count, top1.avg

    def utils_loader(self, data, batch_size, train_or_test):
        x = torch.reshape(torch.tensor(data['x']), (-1, 1, 28, 28))
        y = torch.tensor(data['y'], dtype=torch.int64)
        if train_or_test == 'train':
            batch_size = batch_size
        elif train_or_test == 'test':
            batch_size = batch_size
        torch_dataset = Data.TensorDataset(x, y)
        data_leaf_loader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        return data_leaf_loader
