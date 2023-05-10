from utils.privacy import round_account

class Task:
    def __init__(self, task_id, model_type, round_num, eps0, model):
        self.task_id = task_id
        self.model_type = model_type
        self.round_num = round_num
        self.running_round = 0
        self.eps0 = eps0

        self.model = model
        self.acc = 0.0

        self.available_clients_blocks = {}
        self.clients = []
        self.selected_clients_blocks = {}
        self.model_before = {}
        self.selected_day_blocks = []
        self.max_selected_block_budget = 0.0
        self.trained_round = 0

        self.first_round_eps = round_account(eps0)
        self.demand_total_eps = round_account(eps0, round_num)
