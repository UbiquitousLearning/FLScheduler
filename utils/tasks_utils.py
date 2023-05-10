import numpy as np
import random

def random_tasks(num_tasks, num_rounds):
    N=num_tasks
    R=np.arange(0,num_rounds)
    result = []
    while (N > 0):
        r = random.choice(R)
        np.delete(R, r)
        n = 1
        # n = random.choice(np.arange(1,N+1))
        N = N - n
        result.append((r,n))
    tasks_num = np.zeros(num_rounds, dtype=int)
    for elem in result:
        tasks_num[elem[0]] = elem[1]
    return tasks_num

def random_tasks_two_stage(num_tasks, num_rounds):
    result = []
    N1 = 5
    R1=np.arange(0, 100)
    while (N1 > 0):
        r = random.choice(R1)
        np.delete(R1, r)
        n = 1
        # n = random.choice(np.arange(1,N+1))
        N1 = N1 - n
        result.append((r,n))
    N2 = 5
    R2 = np.arange(3000, 4000)
    while (N2 > 0):
        r = random.choice(R2)
        np.delete(R2, r-3000)
        n = 1
        # n = random.choice(np.arange(1,N+1))
        N2 = N2 - n
        result.append((r,n))
    tasks_num = np.zeros(num_rounds, dtype=int)
    for elem in result:
        tasks_num[elem[0]] = elem[1]
    return tasks_num
