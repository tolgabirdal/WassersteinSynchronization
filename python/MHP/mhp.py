import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import pandas as pd


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