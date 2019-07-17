import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import pandas as pd


def sink(M, reg, numItermax=1000, stopThr=1e-9, cuda = True):

    # we assume that no distances are null except those of the diagonal of
    # distances

    if cuda:
        a = Variable(torch.ones((M.size()[0],)) / M.size()[0]).cuda()
        b = Variable(torch.ones((M.size()[1],)) / M.size()[1]).cuda()
    else:
        a = Variable(torch.ones((M.size()[0],)) / M.size()[0])
        b = Variable(torch.ones((M.size()[1],)) / M.size()[1])

    # init data
    Nini = len(a)
    Nfin = len(b)

    if cuda:
        u = Variable(torch.ones(Nini) / Nini).cuda()
        v = Variable(torch.ones(Nfin) / Nfin).cuda()
    else:
        u = Variable(torch.ones(Nini) / Nini)
        v = Variable(torch.ones(Nfin) / Nfin)

    # print(reg)

    K = torch.exp(-M / reg)
    # print(np.min(K))

    Kp = (1 / a).view(-1, 1) * K
    cpt = 0
    err = 1
    while (err > stopThr and cpt < numItermax):
        uprev = u
        vprev = v
        #print(T(K).size(), u.view(u.size()[0],1).size())
        KtransposeU = K.t().matmul(u)
        v = torch.div(b, KtransposeU)
        u = 1. / Kp.matmul(v)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            transp = u.view(-1, 1) * (K * v)
            # err = (torch.sum(transp) - b).norm(1).pow(2).data[0]
            err = (torch.sum(transp,dim=0) - b).norm(2).pow(2).data.item()


        cpt += 1

    return torch.sum(u.view((-1, 1)) * K * v.view((1, -1)) * M)


def sink_stabilized(M, reg, numItermax=1000, tau=1e2, stopThr=1e-9, warmstart=None, print_period=20, cuda=True):

    if cuda:
        a = Variable(torch.ones((M.size()[0],)) / M.size()[0]).cuda()
        b = Variable(torch.ones((M.size()[1],)) / M.size()[1]).cuda()
    else:
        a = Variable(torch.ones((M.size()[0],)) / M.size()[0])
        b = Variable(torch.ones((M.size()[1],)) / M.size()[1])

    # init data
    na = len(a)
    nb = len(b)

    cpt = 0
    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        if cuda:
            alpha, beta = Variable(torch.zeros(na)).cuda(), Variable(torch.zeros(nb)).cuda()
        else:
            alpha, beta = Variable(torch.zeros(na)), Variable(torch.zeros(nb))
    else:
        alpha, beta = warmstart

    if cuda:
        u, v = Variable(torch.ones(na) / na).cuda(), Variable(torch.ones(nb) / nb).cuda()
    else:
        u, v = Variable(torch.ones(na) / na), Variable(torch.ones(nb) / nb)

    def get_K(alpha, beta):
        return torch.exp(-(M - alpha.view((na, 1)) - beta.view((1, nb))) / reg)

    def get_Gamma(alpha, beta, u, v):
        return torch.exp(-(M - alpha.view((na, 1)) - beta.view((1, nb))) / reg + torch.log(u.view((na, 1))) + torch.log(v.view((1, nb))))

    # print(np.min(K))

    K = get_K(alpha, beta)
    transp = K
    loop = 1
    cpt = 0
    err = 1
    while loop:

        uprev = u
        vprev = v

        # sinkhorn update
        v = torch.div(b, (K.t().matmul(u) + 1e-16))
        u = torch.div(a, (K.matmul(v) + 1e-16))

        # remove numerical problems and store them in K
        if torch.max(torch.abs(u)).data[0] > tau or torch.max(torch.abs(v)).data[0] > tau:
            alpha, beta = alpha + reg * torch.log(u), beta + reg * torch.log(v)

            if cuda:
                u, v = Variable(torch.ones(na) / na).cuda(), Variable(torch.ones(nb) / nb).cuda()
            else:
                u, v = Variable(torch.ones(na) / na), Variable(torch.ones(nb) / nb)

            K = get_K(alpha, beta)

        if cpt % print_period == 0:
            transp = get_Gamma(alpha, beta, u, v)
            err = (torch.sum(transp) - b).norm(1).pow(2).data[0]

        if err <= stopThr:
            loop = False

        if cpt >= numItermax:
            loop = False

        #if np.any(np.isnan(u)) or np.any(np.isnan(v)):
        #    # we have reached the machine precision
        #    # come back to previous solution and quit loop
        #    print('Warning: numerical errors at iteration', cpt)
        #    u = uprev
        #    v = vprev
        #    break

        cpt += 1

    return torch.sum(get_Gamma(alpha, beta, u, v)*M)

def pairwise_distances(x, y, method='l1'):
    n = x.size()[0]
    m = y.size()[0]
    d = x.size()[1]

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    if method == 'l1':
        dist = torch.abs(x - y).sum(2)
    else:
        dist = torch.pow(x - y, 2).sum(2)

    return dist.float()

def mhp_l1(x, y, epsilon=0.95):
    # x shape = b, m * f, n
    # y shape = b, f, n

    batchsize, features, number_vectors = y.size()

    # x shape = b, m, f, n
    # y shape = b, 1, f, n
    x = x.reshape(batchsize, -1, features, number_vectors)
    y = y.unsqueeze(1)

    m = x.size()[1]

    # l1 shape = b, m , n
    l1 = torch.abs(x - y).sum(2)

    # shape = b, n
    best_assignment = l1.min(1)[0]
    all_assignments = l1.sum(1)

    # shape = b, n
    l1_mhp = (epsilon - 1 / m) * best_assignment + (1 - epsilon) * 1 / m * all_assignments

    return l1_mhp

# With hypotheses dropout
def mhp_l1_dropout(x, y, epsilon, dropout=0.5):

    # x shape = b, m * f, n
    # y shape = b, f, n
    batchsize, features, number_vectors = y.size()

    # x shape = b, m, f, n
    # y shape = b, 1, f, n
    x = x.reshape(batchsize, -1, features, number_vectors)
    y = y.unsqueeze(1)

    m = x.size()[1]

    if m > 1:
        num_surv = int((1 - dropout) * m)
        conc = []
        for i in range(batchsize):
            drop_in = torch.randperm(m)[:num_surv]
            conc.append(x.index_select(1, drop_in))

        x = torch.stack(conc, dim=1)

    else:
        num_surv = m

    # l1 shape = b, m - drop , n
    l1 = torch.abs(x - y).sum(2)

    # shape = b, n
    best_assignment = l1.min(1)[0]
    all_assignments = l1.sum(1)

    l1_mhp = (epsilon - 1 / num_surv) * best_assignment + (1 - epsilon) * 1 / num_surv * all_assignments

    return  l1_mhp

def dmat(x,y):
    mmp1 = torch.stack([x] * x.size()[0])
    mmp2 = torch.stack([y] * y.size()[0]).transpose(0, 1)
    mm = torch.sum((mmp1 - mmp2) ** 2, 2).squeeze().cuda()

    return mm   