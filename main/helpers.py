import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import IntEnum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import trange
from collections import defaultdict

from GrU_nn import neuralGrU
from GrU import (evaluate, GrU)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class clock( IntEnum ):
    pv = 0
    storage = 1
    charge = 2
    discharge = 3

class source( IntEnum ):
    grid =  0
    pv = 1
    storage = 2
    charge = 3
    discharge = 4


def getPriceVectors(round):
    '''
    Get price vectors for a round from the clock data.

    Parameters
    ----------
    round : int
        Round number to get the price vectors for, range = [0, 308]

    Returns
    -------
    prices : ndarray of shape (5, 24)
        Price vectors for g, p, s, c, d.
    
    '''
    
    df_auctioneer_data = pd.read_csv('../data/Auctioneer Data.csv')

    headers = np.load(file='../data/clock_data_columns.npy', allow_pickle=True)
    df_clock_data = pd.DataFrame(np.load('../data/clock_data_values.npy', allow_pickle=True), columns=headers)

    prices = np.ndarray((5, 24))
    prices[0] = df_auctioneer_data.GRID_PRICE.values
    for e in clock:
        prices[e + 1] = df_clock_data[f'price_{round}'][e]
    return prices

## LOSS FUNCTION
def total_loss(pi_g, pi_p, pi_s, pi_c, pi_d, d_star):
    '''
    Loss = -revenue
    '''
    # r_g = pi_g * d_star[source.grid]
    r_p = pi_p * d_star[source.pv]
    r_s = pi_s * d_star[source.storage]
    r_c = pi_c * d_star[source.charge]
    r_d = pi_d * d_star[source.discharge]

    r_t = r_p + r_s + r_c + r_d
    rev = torch.sum(r_t)

    return -rev


## REGULARISATION
def barrier_loss(C_g, C, epsilon):
    '''
    Barrier function with barrier at C_g & sensitivity epsilon
    '''
    b = F.relu(-torch.log((C_g - C)/epsilon))

    return torch.sum(b)



def squash( a, scale = 2 ):
    '''
    Exponential min-max scaler
    '''
    m = min( a )
    M = max( a )

    p = np.exp( -scale )
    P = np.exp( scale )

    n = (a - m)/(M-m) * 2 * scale
    n = np.exp( n - scale )

    return (n - p) / (P - p) * (M - m) + m