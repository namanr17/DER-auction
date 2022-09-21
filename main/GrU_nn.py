import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class gru_module_1(nn.Module):
    '''
    Module-1 of Neural GrU.

    Parameters
    ----------
    pi_s : torch tensor of shape (24, )
        Storage prices

    pi_c : torch tensor of shape (24, )
        Charging prices

    pi_g : torch tensor of shape (24, )
        Grid prices

    B : float
        Inverse Temperature

    Returns
    -------
    X2 : torch tensor of shape (24, )
        pi_tilda_c : Price of importing power from cheapest time slot to t
    
    X3 : torch tensor of shape (24, 24)
        i^t : One hot vector representing the cheapest time slot for charging the power to be discharged at time slot t
    '''

    def __init__(self, device, eta_c, eta_d):
        super().__init__()

        self.device = device
        self.eta_c = eta_c
        self.eta_d = eta_d
        self.sum_layer = nn.Linear(in_features=24*3, out_features=276, bias=False)

        # input order = (storage, charge, grid)
        with torch.no_grad():
            W = torch.zeros(size=(276, 72), dtype=float, device=device)

            # for nodes of sum_layer
            j = 0
            for t in range(1, 24):

                # for nodes of input layer
                for i in range(t):
                    W[j, i+24] = 1                              # pi_tau^c
                    W[j, i+48] = 1                              # pi_tau^g
                    W[j, i:t] = 1                               # pi_i^s for i in range(tau, t)
                    j = j + 1

            self.sum_layer.weight = nn.parameter.Parameter(W)


    def forward(self, pi_s, pi_c, pi_g, B):
        pi_c_eta = pi_c / ( self.eta_c * self.eta_d )
        pi_s_eta = pi_s / self.eta_d

        X = torch.cat([pi_s_eta, pi_c_eta, pi_g])

        X1 = self.sum_layer(X)

        X2 = torch.zeros(size=(24,)).to(self.device)
        X3 = torch.zeros(size=(24,24)).to(self.device)

        # Mask out 'inf' using torch.isinf() method
        X2[0] = float('inf')

        X2[1] = X1[0]
        X3[1][0] = 1
        
        idx_nodes = 1
        for t in range(2, 24):
            # Remove unsqueeze to handle batch(s)
            e_t = torch.unsqueeze(X1[idx_nodes:idx_nodes+t], 0)
            X2[t] = -1 * F.max_pool1d(-1 * e_t, kernel_size=t)

            e_t = torch.squeeze(e_t)
            X3[t, :t] = F.softmin(B * e_t, dim=0)

            idx_nodes += t

        return X2, X3



class gru_module_2(nn.Module):
    '''
    Module-2 of Neural GrU.

    Parameters
    ----------
    pi_tilda_c : torch tensor of shape (24, )
        Price of importing power from cheapest time slot
    
    pi_d : torch tensor of shape (24, )
        Discharging prices

    pi_p : torch tensor of shape (24, )
        PV Energy prices

    pi_g : torch tensor of shape (24, )
        Grid prices

    Returns
    -------
    Y : torch tensor of shape (3, 24, 3)
        a^j_t : One hot vectors representing the first (j=0), second (j=1) & third (j=2) cheapest source [d', p, g]; pi_d' = pi_d + pi_tilda_c
    '''

    def __init__(self, eta_d, alpha):
        super().__init__()

        self.eta_d = eta_d
        self.alpha = alpha

    def forward(self, pi_tilda_c, pi_d, pi_p, pi_g, B):
        pi_d_eta = pi_d / self.eta_d

        X = torch.stack([pi_tilda_c + pi_d_eta, pi_p, pi_g], dim=1)

        X1 = F.softmin(2 * B * X, dim=1)
        X2 = F.softmin(2 * B * (X + self.alpha * X1), dim=1)
        X3 = F.softmin(2 * B * (X  + self.alpha * X1 + self.alpha * X2), dim=1)

        Y = torch.stack([X1, X2, X3], dim=0)

        return Y



class gru_module_3(nn.Module):
    '''
    Module-3 of Neural GrU.

    Parameters
    ----------
    X : torch tensor of shape (24, 24)
        i^t : One hot vector representing the cheapest time slot for charging to be discharged at time slot t

    Returns
    -------
    X2 : torch tensor of shape (24, 24)
        i_tilda^t : One hot vector representing the time slots to store the charge to be discharged at time slot t

    '''

    def __init__(self):
        super().__init__()


    def forward(self, X):
        X1 = torch.cumsum(X, dim=1)
        X2 = torch.tril(X1, diagonal=-1)
        return X2



class gru_module_4(nn.Module):
    '''
    Module-4 of Neural GrU.

    Parameters
    ----------
    X : torch tensor of shape (24, )
        d_t : Total demand vector

    X_m1 : torch tensor of shape (24, 24)
        i^t : One hot vector representing the cheapest time slot to import power from

    X_m2 : torch tensor of shape (3, 24, 3)
        a^j_t : One hot vectors representing the first (j=0), second (j=1) & third (j=2) cheapest source [d', p, g]; pi_d' = pi_d + pi_tilda_c

    X_m3 : torch tensor of shape (24, 24)
        i_tilda^t : One hot vector representing the time slots to store the charge to be discharged at time slot t.

    C : torch tensor of shape (24, 3)
        C^i : Constraints on discharging demand (i=0), PV demand (i=1), grid demand (i=2).

    Returns
    -------
    Y : torch tensor of shape (5, 24)
        d^star_t : demand breakup vectors

    '''

    def __init__(self, eta_c, eta_d):
        super().__init__()

        self.eta_c = eta_c
        self.eta_d = eta_d


    def forward(self, X, X_m1, X_m2, X_m3, C):

        a_0, a_1, a_2 = X_m2

        # Distribution for the cheapest source (j=0)
        delta_0 = torch.sum(a_0 * C, dim=1)
        d_0 = delta_0 - F.relu(delta_0 - X)

        # Distribution for second cheapest source (j=1)
        delta_1 = torch.sum(a_1 * C, dim=1)
        d_1 = delta_1 - F.relu(delta_1 - (X - d_0))

        # Distribution for second cheapest source (j=1)
        delta_2 = torch.sum(a_2 * C, dim=1)
        d_2 = delta_2 - F.relu(delta_2 - (X - d_0 - d_1))

        # d_1 = 0
        # d_2 = 0

        d_g = (a_0[:, 2] * d_0) + (a_1[:, 2] * d_1) + (a_2[:, 2] * d_2)
        d_p = (a_0[:, 1] * d_0) + (a_1[:, 1] * d_1) + (a_2[:, 1] * d_2)
        d_d = ((a_0[:, 0] * d_0) + (a_1[:, 0] * d_1) + (a_2[:, 0] * d_2)) / self.eta_d

        d_c = torch.sum(X_m1 * d_d.reshape((24,1)) / self.eta_c, dim=0)
        d_s = torch.sum(X_m3 * d_d.reshape((24,1)), dim=0)

        d_g_c = d_g + d_c

        Y = torch.stack([d_g_c, d_p, d_s, d_c, d_d])

        return Y


class neuralGrU(nn.Module):
    '''
    Neural Network implementation of GrU Algorithm.
    Connects the various modules of Neural GrU.
    '''

    def __init__(self, B, device=None, eta_c=1, eta_d=1, alpha=1e3):
        '''
        @parameter B: Inverse temp for softmin
        @parameter device: torch.device to load the model on
        @parameter eta_c: Charging efficiency, default=1
        @parameter eta_d: Discharging efficiency, default=1
        @parameter alpha: Price multiplier (alpha > max possible price), default=1e3
        '''
        super().__init__()

        self.B = B
        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.m1 = gru_module_1(device=device, eta_c=eta_c, eta_d=eta_d)
        self.m2 = gru_module_2(eta_d=eta_d, alpha=alpha)
        self.m3 = gru_module_3()
        self.m4 = gru_module_4(eta_c=eta_c, eta_d=eta_d)

    def forward(self, pi_g, pi_p, pi_s, pi_c, pi_d, d_t, C_t):

        ppi_p = F.relu(pi_p)
        ppi_s = F.relu(pi_s)
        ppi_c = F.relu(pi_c)
        ppi_d = F.relu(pi_d)

        pi_tilda_c_t, i_t = self.m1(ppi_s, ppi_c, pi_g, self.B)
        a_t = self.m2(pi_tilda_c_t, ppi_d, ppi_p, pi_g, self.B)
        i_tilda_t = self.m3(i_t)
        d_star = self.m4(d_t, i_t, a_t, i_tilda_t, C_t)

        return d_star