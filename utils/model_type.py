from random import choice
from utils.args import parse_args
args = parse_args()
# task_requirments = [['small', 100, 1], ['large', 1000, 2]] ## [model_type, round_num, eps0]/
# task_requirments = [['small', args.small_round_num, args.small_eps0], ['large', args.large_round_num, args.large_eps0]] ## [model_type, round_num, eps0]
# task_requirments = [['small', args.small_round_num, args.small_eps0]] ## [model_type, round_num, eps0]
# task_requirments = [['large', args.large_round_num, args.large_eps0]] ## [model_type, round_num, eps0]

# small:large = 1:1
# def generate_task_requirments(model_mode):
#     task_requirments = []
#     if model_mode == 'small':
#         task_requirments = [['small', args.small_round_num, args.small_eps0]]
#     elif model_mode == 'large':
#         task_requirments = [['large', args.large_round_num, args.large_eps0]]
#     else:
#         task_requirments = [['small', args.small_round_num, args.small_eps0], ['large', args.large_round_num, args.large_eps0]]
#     return choice(task_requirments)

# small:large = 4:1
# def generate_task_requirments(model_mode):
#     task_requirments = []
#     if model_mode == 'small':
#         task_requirments = [['small', args.small_round_num, args.small_eps0]]
#     elif model_mode == 'large':
#         task_requirments = [['large', args.large_round_num, args.large_eps0]]
#     else:
#         task_requirments = [['small', args.small_round_num, args.small_eps0], ['small', args.small_round_num, args.small_eps0], ['small', args.small_round_num, args.small_eps0], ['small', args.small_round_num, args.small_eps0], ['large', args.large_round_num, args.large_eps0]]
#     return choice(task_requirments)

# small:large = 1:4
def generate_task_requirments(model_mode):
    task_requirments = []
    if model_mode == 'small':
        task_requirments = [['small', args.small_round_num, args.small_eps0]]
    elif model_mode == 'large':
        task_requirments = [['large', args.large_round_num, args.large_eps0]]
    else:
        task_requirments = [['small', args.small_round_num, args.small_eps0], ['large', args.large_round_num, args.large_eps0], ['large', args.large_round_num, args.large_eps0], ['large', args.large_round_num, args.large_eps0], ['large', args.large_round_num, args.large_eps0]]
    return choice(task_requirments)

# small:large = 2:3    
# def generate_task_requirments(model_mode):
#     task_requirments = []
#     if model_mode == 'small':
#         task_requirments = [['small', args.small_round_num, args.small_eps0]]
#     elif model_mode == 'large':
#         task_requirments = [['large', args.large_round_num, args.large_eps0]]
#     else:
#         task_requirments = [['small', args.small_round_num, args.small_eps0], ['small', args.small_round_num, args.small_eps0], ['large', args.large_round_num, args.large_eps0], ['large', args.large_round_num, args.large_eps0], ['large', args.large_round_num, args.large_eps0]]
#     return choice(task_requirments)

# small:large = 3:2
# def generate_task_requirments(model_mode):
#     task_requirments = []
#     if model_mode == 'small':
#         task_requirments = [['small', args.small_round_num, args.small_eps0]]
#     elif model_mode == 'large':
#         task_requirments = [['large', args.large_round_num, args.large_eps0]]
#     else:
#         task_requirments = [['small', args.small_round_num, args.small_eps0], ['small', args.small_round_num, args.small_eps0], ['small', args.small_round_num, args.small_eps0], ['large', args.large_round_num, args.large_eps0], ['large', args.large_round_num, args.large_eps0]]
#     return choice(task_requirments)
