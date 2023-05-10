import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--init-eps',
                    help='total budget of each initial block;',
                    type=float,
                    default=10.0)
    parser.add_argument('--delta',
                    help='Probability of violate privacy restrictions;',
                    type=float,
                    default=1e-6)
    parser.add_argument('--client-count',
                    help='the number of total client;',
                    type=int,
                    default=10000)
    parser.add_argument('--l2-norm-clip-LeNet',
                    help='Clipping norm;',
                    type=float,
                    default=0.005)
    parser.add_argument('--lr-LeNet',
                    help='learning rate;',
                    type=float,
                    default=0.01)
    parser.add_argument('--l2-norm-clip-ResNet',
                    help='Clipping norm;',
                    type=float,
                    default=0.005)
    parser.add_argument('--lr-ResNet',
                    help='learning rate;',
                    type=float,
                    default=0.01)

    parser.add_argument('--round-schedule',
                    help='total round to schedule;',
                    type=int,
                    default=4000)
    parser.add_argument('--lam',
                    help='task arrival rate;',
                    type=int,
                    default=2)
    parser.add_argument('--total-tasks',
                    help='number of total tasks;',
                    type=int,
                    default=10)

    parser.add_argument('--block-size',
                    help='the number of data per block;',
                    type=int,
                    default=5)
    parser.add_argument('--block-count',
                    help='the number of total block in a client;',
                    type=int,
                    default=10)

    parser.add_argument('--batch-size',
                    help='batch size;',
                    type=int,
                    default=5)

    parser.add_argument('--num-clients',
                    help='number of selected clients for each task;',
                    type=int,
                    default=100)
    parser.add_argument('--num-blocks',
                    help='number of selected blocks in each client;',
                    type=int,
                    default=2)
    parser.add_argument('--small-round-num',
                    help='round number required by small task;',
                    type=int,
                    default=2000)
    parser.add_argument('--small-eps0',
                    help='LDP parameter eps0 required by small task;',
                    type=float,
                    default=1)
    parser.add_argument('--large-round-num',
                    help='round number required by large task;',
                    type=int,
                    default=1000)
    parser.add_argument('--large-eps0',
                    help='LDP parameter eps0 required by large task;',
                    type=float,
                    default=2.5)

    parser.add_argument('--total-round',
                    help='total round of scheduling and training;',
                    type=int,
                    default=6000)
    parser.add_argument('--test-round',
                    help='round of testing;',
                    type=int,
                    default=100)

    parser.add_argument('--model-mode',
                    help='model mode of task, including small, large, and hybrid;',
                    type=str,
                    default='large')

    parser.add_argument('--cuda',
                    help='the selected cuda device;',
                    type=str,
                    default='cuda:0')
    return parser.parse_args()
