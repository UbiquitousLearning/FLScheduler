import math
import numpy as np
import scipy.special as sp
from utils.args import parse_args

args = parse_args()

def stable_logsumexp(x):
    a = np.max(x)
    return a+np.log(np.sum(np.exp(x-a)))

def log_comb(n, k):
    return (sp.gammaln(n + 1) -sp.gammaln(k + 1) - sp.gammaln(n - k + 1))
########################################################################################################################
######################################## RDP with Sampling #############################################################
########################################################################################################################
def RDP_comp_integer_order_samp(eps0, n, q, lambd):
    if lambd == 1:
        D = 0
    else:
        ex_eps0 = np.exp(eps0)
        m = (q*n-1)/(2*ex_eps0)+1
        c = 2*((ex_eps0-1/ex_eps0)**2)/(m)
        a = []
        a.append(0)
        term = log_comb(lambd,2)+2*np.log(q)+np.log(((ex_eps0-1)**2)/(m*ex_eps0))+np.log(4)
        a.append(term)
        for j in range(3,lambd+1,1):
            term = log_comb(lambd, j)+np.log(j)+sp.gammaln(j/2)+(j/2)*np.log(c)+j*np.log(q)
            a.append(term)
        a.append(-(q*n-1)/(8*ex_eps0)+np.log(((1+q*(np.exp(eps0)-np.exp(-eps0)))**lambd)-1-q*lambd*(np.exp(eps0)-np.exp(-eps0))))
        D = stable_logsumexp(a)/(lambd-1)
    return D

def RDP_comp_samp(eps0, n, q, lambd):
    D = np.zeros_like(lambd,dtype=float)
    for i in range(0,len(lambd),1):
        c = int(np.ceil(lambd[i]))
        f = int(np.floor(lambd[i]))
        if c==f:
            D[i] = RDP_comp_integer_order_samp(eps0, n, q, f)
        else:
            a = c-lambd[i]
            D[i] = (f-1)*a*RDP_comp_integer_order_samp(eps0, n, q, f)
            D[i] += (c-1)*(1-a)*RDP_comp_integer_order_samp(eps0, n, q, c)
            D[i] /= (lambd[i]-1)
    return D
########################################################################################################################
################################## 1st Upper bound Theorem1 ############################################################
########################################################################################################################
def RDP_comp_integer_order(eps0,n,lambd):
    if lambd == 1:
        D = 0
    else:
        ex_eps0 = np.exp(eps0)
        m = (n-1)/(2*ex_eps0)+1
        c = ((ex_eps0-1/ex_eps0)**2)/(2*m)
        a = []
        a.append(0)
        term = np.log(((ex_eps0-1)**2)/(m*ex_eps0))+log_comb(lambd,2)
        a.append(term)
        for j in range(3,lambd+1,1):
            term = log_comb(lambd, j)+np.log(j)+sp.gammaln(j/2)+(j/2)*np.log(c) 
            a.append(term)
        a.append(-(n-1)/(8*ex_eps0)+eps0*lambd)
        D = stable_logsumexp(a)/(lambd-1)
    return D

def RDP_comp(eps0,n,lambd):
    D = np.zeros_like(lambd,dtype=float)
    for i in range(0,len(lambd),1):
        c = int(np.ceil(lambd[i]))
        f = int(np.floor(lambd[i]))
        if c==f:
            D[i] = RDP_comp_integer_order(eps0,n,f)
        else:
            a = c-lambd[i]
            D[i] = (f-1)*a*RDP_comp_integer_order(eps0,n,f)
            D[i] += (c-1)*(1-a)*RDP_comp_integer_order(eps0,n,c)
            D[i] /= (lambd[i]-1)
    return D
########################################################################################################################
################################# Lower bound Theorem 3 ################################################################
########################################################################################################################
def lower_bound_RDP(eps0,n,q,lambd):
    D = np.zeros_like(lambd,dtype=float)
    ex_eps0 = np.exp(eps0)
    p = 1/(ex_eps0+1)
    m = int(q*n)
    k = np.arange(0,m+1,1)
    comb = [log_comb(m,j) for j in k]
    for i in range(0,len(lambd),1):
        a = comb + k*np.log(p) + (m-k)*np.log(1-p) 
        a += lambd[i]*np.log((1-q)+q*(k*(ex_eps0-1/ex_eps0)/m+1/ex_eps0))
        D[i] = stable_logsumexp(a)
        D[i]/= lambd[i]-1
    return D
########################################################################################################################
############################################ Uniform Sub-Sampling ######################################################
########################################################################################################################
def uniform_subsampled_integer_order(eps0, n, q, lambd, rdp_fun):
    if lambd == 1:
        D = 0
    else:
        a = []
        a.append(0)
        m = int(q*n)
        D = rdp_fun(eps0,m,np.arange(2,lambd+1,1))
        term = log_comb(lambd,2)+2*np.log(q)+np.log(2)+ D[0]
        a.append(term)
        for j in range(3,lambd+1,1):
            term = log_comb(lambd, j)+j*np.log(q)+np.log(2)+(j-1)*D[j-2]
            a.append(term)
        D = stable_logsumexp(a)/(lambd-1)
    return D

def uniform_subsampled(eps0, n, q, lambd, rdp_fun):
    D = np.zeros_like(lambd,dtype=float)
    for i in range(0,len(lambd),1):
        c = int(np.ceil(lambd[i]))
        f = int(np.floor(lambd[i]))
        if c==f:
            D[i] = uniform_subsampled_integer_order(eps0, n, q, f, rdp_fun)
        else:
            a = c-lambd[i]
            D[i] = (f-1)*a*uniform_subsampled_integer_order(eps0, n, q, f, rdp_fun)
            D[i] += (c-1)*(1-a)*uniform_subsampled_integer_order(eps0, n, q, c, rdp_fun)
            D[i] /= (lambd[i]-1)
    return D
########################################################################################################################
############################################ Optimize from RDP to DP ###################################################
########################################################################################################################
def optimize_RDP_To_DP(delta,acc,eps0,n,q,T,rdp_fun):
    if eps0>1:
        lmax = np.array(10**4)
    else:
        lmax = np.array(10**5)
    lmin = np.array(1)
    err = lmax
    while (err>acc):
        l = []
        l.append((lmax+lmin)/2)
        l.append((lmax+lmin)/2+0.01)
        D = T*rdp_fun(eps0,n,q,l)+np.log(1-1/np.array(l))-np.log(delta*np.array(l))/(np.array(l)-1)
        err = lmax-lmin
        if D[0]>D[1]:
            lmin = l[0]
            eps = D[1]
        else:
            lmax = l[0]
            eps = D[0]
    return eps
########################################################################################################################
############################################ Optimize from RDP to DP ###################################################
########################################################################################################################
def optimize_RDP_To_DP_2(delta,acc,eps0,n,q,T,rdp_fun,rdp_fun2):
#     lambd =[2,4,8,16,32,64,256,512,700,1024,1500,2048,3000,4096,6000,8192,10000]
#     D = T*rdp_fun(eps0,n,q,lambd,rdp_fun2)+np.log(1-1/np.array(lambd))-np.log(delta*np.array(lambd))/(np.array(lambd)-1)
#     eps = np.amin(D)
    if eps0>1:
        lmax = np.array(10**3)
    else:
        lmax = np.array(10**3)
    lmin = np.array(1)
    err = lmax
    while (err>acc):
        l = []
        l.append((lmax+lmin)/2)
        l.append((lmax+lmin)/2+0.01)
        D = T*rdp_fun(eps0,n,q,l,rdp_fun2)+np.log(1-1/np.array(l))-np.log(delta*np.array(l))/(np.array(l)-1)
        err = lmax-lmin
        if D[0]>D[1]:
            lmin = l[0]
            eps = D[1]
        else:
            lmax = l[0]
            eps = D[0]
    return eps

def __init__():
    global f
    f = {}
    with open('eps.txt', 'r') as file:
        for _ in range(2):
            eps = [0.0]
            eps0 = eval(file.readline())
            round = eval(file.readline())
            for i in range(1, round + 1):
                eps.append(eval(file.readline()))
            f[eps0] = eps

def round_account(eps0, round_count = 1):
    return f[eps0][round_count]

def task_account(tasks_eps):
    eps = 0
    for task_eps in tasks_eps:
        eps = eps + task_eps
    return eps
