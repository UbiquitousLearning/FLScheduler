from server import Server
import utils.privacy as privacy
from utils.args import parse_args

args = parse_args()

def main():
    privacy.__init__()
    server = Server()

    import time
    start = time.time()
    server.setup_clients('MNIST', model=None)
    clients_info = server.get_clients_info()
    print("------------read data and setup clients------------------")
    print('total clients: %d, and created blocks of client 1: %d' % (len(clients_info), len(clients_info[1])))
    
    server.generate_tasks()
    tasks_eps_info = server.get_tasks_eps_info()
    server.setup_days()
    print("------------generate task through possion----------------")
    print('total tasks: %d' % server.total_tasks_num)
    print("------------LDP-FLSchedule----------------")

    print("total eps: %d, select clients: %d, select blocks: %d, scheduling round number: %d" % 
        (args.init_eps, args.num_clients, args.num_blocks, args.round_schedule))
    
    for i in range(args.total_round):
        print("------------currrent round number: %d----------------" % i)
        print(f'waiting_tasks = {len(server.waiting_tasks)}, running_tasks = {len(server.running_tasks)}, delayed_tasks = {len(server.delayed_tasks)}, completed_tasks = {len(server.completed_tasks)}')
        server.schedule(current_round=i)

        if i >= args.round_schedule and not server.running_tasks:
            server.schedule_delayed()
    
    completed_task_count, completed_task_avgacc, total_task_avgacc = server.print_accuracy()
    print(f'completed_task_count = {completed_task_count}, completed_task_avgacc = {completed_task_avgacc}, total_task_avgacc = {total_task_avgacc}')
    print("total time {}.".format(time.time() - start))
    print('Hello LDP-FLSchedule')

if __name__ == '__main__':
    main()
