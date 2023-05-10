from utils.args import parse_args
from utils.privacy import round_account, task_account

args = parse_args()

class SmallDay():
    def __init__(self, eps0):
        self.eps0 = eps0
        self.allocated_count = 0
        self.consumed_count = 0

    def allocate_eps(self, count):
        self.allocated_count = self.allocated_count + count

    def consume_eps(self, count):
        self.allocated_count = self.allocated_count - count
        self.consumed_count = self.consumed_count + count
        assert self.allocated_count >= 0, 'Today task cannot consume so much budget'

    def allocated_query(self):
        return round_account(self.eps0, self.consumed_count + self.allocated_count)

    def consumed_query(self):
        return round_account(self.eps0, self.consumed_count)

    def consumed_query_next(self):
        return round_account(self.eps0, self.consumed_count + 1) - round_account(self.eps0, self.consumed_count)

class Day():
    def __init__(self, day_id, tasks):
        self.day_id = day_id
        self.small_days = []

        self.init_eps = args.init_eps
        self.tasks_count = len(tasks)
        self.unlocked_rate = 1.0 / self.tasks_count
        self.unlocked_count = 0

        for task in tasks:
            self.small_days.append(SmallDay(task.eps0))
    
    def unlock_eps(self, count = 1): # locked -> unlocked
        if self.unlocked_count + count > self.tasks_count:
            return False
        self.unlocked_count = self.unlocked_count + count
        return True

    def allocate_eps(self, task, count = 1): # unlocked -> allocated
        self.small_days[task.task_id-1].allocate_eps(count)
    
    def recycle_eps(self, task, count = 1): # allocated -> unlocked
        self.small_days[task.task_id-1].allocate_eps(-count)

    def consume_eps(self, task, count = 1): # allocated -> consumed
        self.small_days[task.task_id-1].consume_eps(count)
        assert self.available_eps() >= 0.0, "This block can't consume too much budget"

    def available_eps(self):
        allocated_eps = []
        for small_day in self.small_days:
            allocated_eps.append(small_day.allocated_query())
        return self.unlocked_rate * self.unlocked_count * self.init_eps - task_account(allocated_eps)

    def consumed_query_next(self, task):
        return self.small_days[task.task_id-1].consumed_query_next()
    
    def consumed_check_next(self, task):
        return self.small_days[task.task_id-1].consumed_query_next() <= self.available_eps()
