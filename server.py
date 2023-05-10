import os
import copy
import collections
import operator
from functools import reduce
import numpy as np

from data.MNIST.model.lenet import lenet
from data.MNIST.model.resnet import ResNet
from client import Client
from task import Task
from day import Day

from utils.args import parse_args
from utils.model_type import generate_task_requirments
from utils.model_utils import read_data
from utils.tasks_utils import random_tasks

args = parse_args()

class Server:
    def __init__(self):
        self.tasks = []
        self.clients = []
        self.days = []
        self.waiting_tasks = []
        self.running_tasks = []
        self.delayed_tasks = []
        self.completed_tasks = []

        self.total_tasks_num = 0
        self.select_seed = 0
    
    def setup_clients(self, dataset, model=None):
        """Instantiates clients based on given train and test data directories.

        return:
            all_clients: list of Client objects
        
        """
        train_data_dir = os.path.join('data', dataset, 'data', 'train')
        test_data_dir = os.path.join('data', dataset, 'data', 'test')

        users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)

        self.clients  = [Client(i+1, train_data[users[i]], test_data[users[i]]) for i in range(len(users))]

    def setup_days(self):
        tasks = []
        for task in self.tasks:
            tasks.extend(task)
        self.days = [Day(i, tasks) for i in range(1, args.block_count+1)]

    def generate_tasks(self):
        tasks_total = []
        tasks_num = random_tasks(args.total_tasks, 4000)

        self.total_tasks_num = np.sum(tasks_num)
        task_index = 0
        for task_num in tasks_num:
            tasks_current = []
            for i in range(task_num):
                task_requirments = generate_task_requirments(args.model_mode)
                model = lenet() if task_requirments[0] == 'small' else ResNet()
                tasks_current.append(Task(task_id=task_index+1, model_type=task_requirments[0], round_num=task_requirments[1], eps0=task_requirments[2], model=model))
                task_index+=1
            tasks_total.append(tasks_current)
        
        self.tasks = tasks_total

    def unlock_eps(self):
        for day in self.days:
            day.unlock_eps()

    def allocate_eps(self, task, round_num = -1):
        for day in self.days:
            day.allocate_eps(task, task.round_num if round_num == -1 else round_num)

    def recycle_eps(self, task):
        blocks_id = []
        for block_id in range(1, args.block_count+1):
            if block_id not in task.selected_day_blocks:
                blocks_id.append(block_id)

        for day_id in blocks_id:
            self.days[day_id-1].recycle_eps(task)

    def consume_eps(self, task):
        for day_id in task.selected_day_blocks:
            self.days[day_id-1].consume_eps(task)

    def min_available_eps(self):
        min_available_eps = args.init_eps
        for day in self.days:
            min_available_eps = min(min_available_eps, day.available_eps())
        # print(f'Minimum available_eps in all blocks = {min_available_eps}')
        return min_available_eps

    def max_demand_next_round_eps(self, task):
        max_demand_next_round_eps = 0
        for client in self.clients:
            max_demand_next_round_eps = max(max_demand_next_round_eps, max(client.demand_eps(task)))
        # print("max demand_next_round_eps in all blocks: %f" % max_demand_next_round_eps)
        return max_demand_next_round_eps
    
    def schedule(self, current_round):
        if current_round < args.round_schedule: 
            for task in self.tasks[current_round]:
                print("task %d arrive, demand_total_eps: %f, model_type: %s" % (task.task_id, task.demand_total_eps, task.model_type))
                task.acc = self.test_model(task, self.clients)
                print('task_id: %d, trained_round: %d, model_acc: %f, model_type: %s' % (task.task_id, task.trained_round, task.acc, task.model_type))

                self.unlock_eps()
                self.waiting_tasks.append(task)
                self.waiting_tasks.sort(key=lambda x:x.demand_total_eps,reverse=False)
                
                tmp_tasks = []
                for task in self.waiting_tasks:
                    print("minimum demand_total_eps: %f, and its task id: %d" % (task.demand_total_eps, task.task_id))
                    min_available_eps = self.min_available_eps()
                    print("minimum available_eps: %f" % (min_available_eps))
                    if min_available_eps >= task.demand_total_eps:
                        print("task %d join running tasks, allocate eps, and remove from waiting tasks" % task.task_id)
                        self.running_tasks.append(task)
                        self.allocate_eps(task)
                        tmp_tasks.append(task)
                    else:
                        break
                for task in tmp_tasks:
                    self.waiting_tasks.remove(task)
                # print("current round: %d, running_tasks: %d, waiting_tasks: %d" % 
                #         (current_round, len(self.running_tasks), len(self.waiting_tasks)))

        else:
            tmp_tasks = []
            for task in self.waiting_tasks:
                print("minimum demand_total_eps: %f, and its task id: %d" % (task.demand_total_eps, task.task_id))
                if self.min_available_eps() >= task.demand_total_eps:
                    print("task %d join running tasks, allocate eps, and remove from waiting tasks" % task.task_id)
                    self.running_tasks.append(task)
                    self.allocate_eps(task)
                    tmp_tasks.append(task)
                else:
                    break
            for task in tmp_tasks:
                self.waiting_tasks.remove(task)
            # print("current round: %d, running_tasks: %d, waiting_tasks: %d" % 
            #         (current_round, len(self.running_tasks), len(self.waiting_tasks)))

        self.fl_train()

    def fl_train(self):
        for task in self.running_tasks:
            if task.trained_round < task.round_num:
                updates = []
                self.select_clients_and_blocks(task)
                model_before = task.model.to(args.cuda)
                task.model_before = copy.deepcopy(model_before.state_dict())
                for client_id in list(task.selected_clients_blocks.keys()):
                    num_samples, grads = self.clients[client_id-1].train(task, task.selected_clients_blocks[client_id])
                    updates.append((num_samples, grads))
                self.update_model(task, updates)
                task.trained_round += 1
                if task.trained_round % args.test_round == 0 or task.trained_round == task.round_num:
                    task.acc = self.test_model(task, self.clients)
                    print('task_id: %d, round: %d, model_acc: %f' % (task.task_id, task.trained_round, task.acc))
                self.consume_eps(task)
                self.recycle_eps(task)
            else:
                print("task %d completed, and remove from running tasks" % task.task_id)
                self.completed_tasks.append(task)
                self.running_tasks.remove(task)

    def schedule_delayed(self):
        if not self.delayed_tasks and self.waiting_tasks:
            task = self.waiting_tasks[0]
            print("minimum first_round_eps: %f, and its task id: %d" % (task.first_round_eps, task.task_id))
            if self.min_available_eps() >= task.first_round_eps:
                print("task %d join delayed tasks, and remove from waiting tasks" % task.task_id)
                self.delayed_tasks.append(task)
                self.waiting_tasks.remove(task)
        
        self.fl_train_delayed()

    def fl_train_delayed(self):
        for task in self.delayed_tasks:
            if task.trained_round < task.round_num and self.consumed_check_next(task):
                self.allocate_eps(task, 1)
                updates = []
                self.select_clients_and_blocks(task)
                model_before = task.model.to(args.cuda)
                task.model_before = copy.deepcopy(model_before.state_dict())
                for client_id in list(task.selected_clients_blocks.keys()):
                    num_samples, grads = self.clients[client_id-1].train(task, task.selected_clients_blocks[client_id])
                    updates.append((num_samples, grads))
                self.update_model(task, updates)
                task.trained_round += 1
                if task.trained_round % args.test_round == 0 or task.trained_round == task.round_num:
                    task.acc = self.test_model(task, self.clients)
                    print('task_id: %d, round: %d, model_acc: %f' % (task.task_id, task.trained_round, task.acc))
                self.consume_eps(task)
                self.recycle_eps(task)
            elif task.trained_round >= task.round_num:
                print("task %d completed, and remove from delayed tasks" % task.task_id)
                self.completed_tasks.append(task)
                self.delayed_tasks.remove(task)

    def consumed_check_next(self, task):
        for day in self.days:
            if day.consumed_check_next(task) == False:
                return False
        return True

    def get_demand_next_round(self, task):
        eps_info = {}
        for day in self.days:
            eps_info[day.day_id] = day.consumed_query_next(task)
        return eps_info

    def get_availabe_budget(self):
        eps_info = {}
        for day in self.days:
            eps_info[day.day_id] = day.available_eps()
        return eps_info

    def update_model(self, task, updates):
        total_weight = 0.
        base = [0] * len(updates[0][1])
        for (client_samples, client_grads) in updates:
            total_weight += client_samples
            for i, v in enumerate(client_grads):
                base[i] += (client_samples * v)
        averaged_grad = [v / total_weight for v in base]

        before = task.model_before
        base = [0] * len(averaged_grad)
        for i, v in enumerate(list(task.model_before.values())):
            base[i] = v.cpu() - averaged_grad[i].cpu()

        layer_name_list = list(task.model_before.keys())
        state_dict_agg = self.construct_dict(base=base, layer_name_list=layer_name_list)
        task.model.load_state_dict(state_dict_agg)
    
        after = task.model.state_dict()

    def construct_dict(self, base, layer_name_list):
        state_dict_agg=collections.OrderedDict()

        layer_index = 0
        for weight in base:
            state_dict_agg[layer_name_list[layer_index]] = weight
            layer_index += 1

        return state_dict_agg

    def test_model(self, task, clients_to_test):
        metrics = {}
        for client in clients_to_test:
            if client.client_id != 1:
                break
            test_samples, acc = client.test(task)
            metrics[client.client_id] = (test_samples,acc)
        total_acc = 0.0
        total_samples = 0
        for element in metrics.values():
            total_samples += element[0]
            total_acc += (element[0] * element[1])
        avg_acc = total_acc / total_samples
        return avg_acc

    def select_clients_and_blocks(self, task):
        selected_clients_blocks = {} 
        selected_day_blocks = np.random.choice([i+1 for i in range(args.block_count)], args.num_blocks, replace=False)
        task.selected_day_blocks = selected_day_blocks
        selected_clients = np.random.choice([client.client_id for client in self.clients], args.num_clients, replace=False)
        for selected_client in selected_clients:
            selected_clients_blocks[selected_client] = selected_day_blocks
        task.selected_clients_blocks = selected_clients_blocks
    
    def print_accuracy(self):
        EPS = 1e-10
        total_task = 0
        completed_task = 0
        total_acc = 0.0
        for task in reduce(operator.add, self.tasks):
            print('task %d finish, model_acc: %f, trained_round: %d, model_type: %s, ' % (task.task_id, task.acc, task.trained_round, task.model_type))
            print('task_id: %d, model_acc: %f' % (task.task_id, task.acc))
            total_task = total_task + 1
            if not(task.acc < EPS):
                completed_task = completed_task + 1
                total_acc = total_acc + task.acc
        completed_task_count = completed_task
        completed_task_avgacc = total_acc / completed_task if completed_task != 0 else 0.0
        total_task_avgacc = total_acc / total_task if total_task != 0 else 0.0
        return completed_task_count, completed_task_avgacc, total_task_avgacc

    def get_clients_info(self):
        """Return {client_id: (num_samples, [block_id])}
        """
        info_dict = {}
        for client in self.clients:
            info_dict[client.client_id] =  [b.block_id for b in client.blocks]
        return info_dict

    def get_tasks_eps_info(self):
        info_dict = {}
        for task in reduce(operator.add, self.tasks):
            info_dict[task.task_id] = (task.demand_total_eps, task.first_round_eps)
        return info_dict
