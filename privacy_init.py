from tqdm import trange
from utils.privacy import RDP_comp_samp, optimize_RDP_To_DP
from utils.args import parse_args

args = parse_args()

def main():
    q = 1.0 * args.num_clients / args.client_count
    with open('eps-2.txt', 'w') as file:
        eps0 = args.small_eps0
        round = args.small_round_num
        print(eps0, file=file)
        print(round, file=file)
        for i in trange(1, round + 1):
            print(optimize_RDP_To_DP(args.delta, 0.01, eps0, args.client_count, q, i, RDP_comp_samp), file=file)

        eps0 = args.large_eps0
        round = args.large_round_num
        print(eps0, file=file)
        print(round, file=file)
        for i in trange(1, round + 1):
            print(optimize_RDP_To_DP(args.delta, 0.01, eps0, args.client_count, q, i, RDP_comp_samp), file=file)

if __name__ == '__main__':
    main()
