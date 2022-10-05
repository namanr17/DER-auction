import torch
import torch.nn as nn
import torch.nn.functional as F


class gru_module_1(nn.Module):
    '''
    Module-1 of Neural GrU.

    Attributes
    ----------
    device : torch.device type, default=None (cpu)
        Device to load the model on

    eta_c : float type, default=1
        Charging efficiency

    eta_d : float type, default=1
        Discharging efficiency
    '''
    
    def __init__(self, n_agents, device, eta_c, eta_d):
        super().__init__()

        self.n_agents = n_agents
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
        '''
        Forward method.
        
        Parameters
        ----------
        pi_s : torch tensor of shape (n_agents, 24)
            Storage prices

        pi_c : torch tensor of shape (n_agetns, 24)
            Charging prices

        pi_g : torch tensor of shape (n_agents, 24)
            Grid prices

        B : float
            Inverse Temperature

        Returns
        -------
        X2 : torch tensor of shape (n_agents, 24)
            pi_tilda_c : Price of importing power from cheapest time slot to t
        
        X3 : torch tensor of shape (n_agents, 24, 24)
            i^t : One hot vector representing the cheapest time slot for charging the power to be discharged at time slot t
        '''
     
        pi_c_eta = pi_c / ( self.eta_c * self.eta_d )
        pi_s_eta = pi_s / self.eta_d

        X = torch.cat([pi_s_eta, pi_c_eta, pi_g], dim=1)

        X1 = self.sum_layer(X)

        X2 = torch.zeros(size=(self.n_agents, 24)).to(self.device)
        X3 = torch.zeros(size=(self.n_agents, 24, 24)).to(self.device)

        # Mask out 'inf' using torch.isinf() method
        X2[:, 0] = float('inf')

        X2[:, 1] = X1[:, 0]
        X3[:, 1, 0] = 1
        
        idx_nodes = 1
        for t in range(2, 24):
            e_t = X1[:, idx_nodes:idx_nodes+t]

            X2[:, t] = -1 * F.max_pool1d(-1 * e_t, kernel_size=t).T
            X3[:, t, :t] = F.softmin(B * e_t, dim=1)

            idx_nodes += t

        return X2, X3



class gru_module_2(nn.Module):
    '''
    Module-2 of Neural GrU.

    Attributes
    ----------
    eta_d : float type
        Discharging efficiency
    
    alpha : float type
        Price multiplier (alpha > max possible price)
    '''

    def __init__(self, eta_d, alpha):
        super().__init__()

        self.eta_d = eta_d
        self.alpha = alpha

    def forward(self, pi_tilda_c, pi_d, pi_p, pi_g, B):
        '''
        Forward method.

        Parameters
        ----------
        pi_tilda_c : torch tensor of shape (n_agetns, 24)
            Price of importing power from cheapest time slot
        
        pi_d : torch tensor of shape (n_agents, 24)
            Discharging prices

        pi_p : torch tensor of shape (n_agents, 24)
            PV Energy prices

        pi_g : torch tensor of shape (n_agents, 24)
            Grid prices

        B : float type
            Inverse temperature for Softmin

        Returns
        -------
        X1 : torch tensor of shape (n_agents, 24, 3)
            a^1_t : One hot vectors representing the first cheapest source [d', p, g]; pi_d' = pi_d + pi_tilda_c
        
        X2 : torch tensor of shape (n_agents, 24, 3)
            a^2_t : One hot vectors representing the second cheapest source [d', p, g]; pi_d' = pi_d + pi_tilda_c

        X3 : torch tensor of shape (n_agents, 24, 3)
            a^3_t : One hot vectors representing the third cheapest source [d', p, g]; pi_d' = pi_d + pi_tilda_c
        '''

        pi_d_eta = pi_d / self.eta_d

        X = torch.stack([pi_tilda_c + pi_d_eta, pi_p, pi_g], dim=2)

        X1 = F.softmin(2 * B * X, dim=2)
        X2 = F.softmin(2 * B * (X + self.alpha * X1), dim=2)
        X3 = F.softmin(2 * B * (X  + self.alpha * X1 + self.alpha * X2), dim=2)

        return (X1, X2, X3)



class gru_module_3(nn.Module):
    '''
    Module-3 of Neural GrU.
    '''

    def __init__(self):
        super().__init__()


    def forward(self, X):
        '''
        Forward method.

        Parameters
        ----------
        X : torch tensor of shape (n_agents, 24, 24)
            i^t : One hot vector representing the cheapest time slot for charging to be discharged at time slot t

        Returns
        -------
        X2 : torch tensor of shape (n_agents, 24, 24)
            i_tilda^t : One hot vector representing the time slots to store the charge to be discharged at time slot t

        '''
        X1 = torch.cumsum(X, dim=2)
        X2 = torch.tril(X1, diagonal=-1)
        return X2



class gru_module_4(nn.Module):
    '''
    Module-4 of Neural GrU.

    Attributes
    ----------
    n_agents : int, default=1
        Number of agents

    eta_c : float type
        Charging efficiency

    eta_d : float type
        Discharging efficiency

    '''

    def __init__(self, n_agents, eta_c, eta_d):
        super().__init__()

        self.n_agents = n_agents
        self.eta_c = eta_c
        self.eta_d = eta_d


    def forward(self, X, X_m1, X_m2, X_m3, C):
        '''
        Forward method.
        
        Parameters
        ----------
        X : torch tensor of shape (n_agents, 24)
            d_t : Total demand vector

        X_m1 : torch tensor of shape (n_agents, 24, 24)
            i^t : One hot vector representing the cheapest time slot to import power from

        X_m2 : Tuple of torch tensors of shapes (n_agents, 24, 3)
            a^j_t : One hot vectors representing the first (j=0), second (j=1) & third (j=2) cheapest source [d', p, g]; pi_d' = pi_d + pi_tilda_c

        X_m3 : torch tensor of shape (n_agents, 24, 24)
            i_tilda^t : One hot vector representing the time slots to store the charge to be discharged at time slot t.

        C : torch tensor of shape (n_agents, 24, 3)
            C^i : Constraints on discharging demand (i=0), PV demand (i=1), grid demand (i=2).

        Returns
        -------
        Y : torch tensor of shape (5, 24)
            d^star_t : demand breakup vectors
        '''

        a_0, a_1, a_2 = X_m2

        # Distribution for the cheapest source (j=0)
        delta_0 = torch.sum(a_0 * C, dim=2)
        d_0 = delta_0 - F.relu(delta_0 - X)

        # Distribution for second cheapest source (j=1)
        delta_1 = torch.sum(a_1 * C, dim=2)
        d_1 = delta_1 - F.relu(delta_1 - (X - d_0))

        # Distribution for second cheapest source (j=1)
        delta_2 = torch.sum(a_2 * C, dim=2)
        d_2 = delta_2 - F.relu(delta_2 - (X - d_0 - d_1))

        # d_1 = 0
        # d_2 = 0

        d_g = (a_0[:, :, 2] * d_0) + (a_1[:, :, 2] * d_1) + (a_2[:, :, 2] * d_2)
        d_p = (a_0[:, :, 1] * d_0) + (a_1[:, :, 1] * d_1) + (a_2[:, :, 1] * d_2)
        d_d = (a_0[:, :, 0] * d_0) + (a_1[:, :, 0] * d_1) + (a_2[:, :, 0] * d_2)

        d_c = torch.sum(X_m1 * d_d.reshape((self.n_agents, 24,1)) / self.eta_c, dim=1)
        d_s = torch.sum(X_m3 * d_d.reshape((self.n_agents, 24,1)), dim=1)

        d_g_c = d_g + d_c

        Y = torch.stack([d_g_c, d_p, d_s, d_c, d_d], dim=1)

        return Y


class neuralGrU(nn.Module):
    '''
    Neural Network implementation of GrU Algorithm.
    Connects the various modules of Neural GrU.

    Attributes
    ----------
    B : float
        Inverse Temperature

    n_agents : int, default=1
        Number of agents

    ep : ndarray of shape (24, )
        Epsilon values for Environmental Impact
    
    gamma : ndarray of shape (n_agents, )
        Gamma values of EI for all agents

    device : torch.device type, default=None (cpu)
        Device to load the model on

    eta_c : float type, default=1
        Charging efficiency

    eta_d : float type, default=1
        Discharging efficiency
    
    alpha : float type, default=1e3
        Price multiplier (alpha > max possible price)
    '''

    def __init__(self, B, n_agents=1, ep=None, gamma=None, device=None, eta_c=1, eta_d=1, alpha=1e3):
        super().__init__()

        self.B = B
        self.n_agents = n_agents
        self.device = device
        self.alpha = alpha
        self.eta_c = eta_c
        self.eta_d = eta_d

        self.gamma = torch.tensor(gamma, requires_grad=False, device=device).unsqueeze(dim=0)
        self.ep_t_n = torch.tensor(ep, requires_grad=False, device=device).expand(n_agents, -1) * self.gamma.T

        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.m1 = gru_module_1(n_agents, device, eta_c, eta_d)
        self.m2 = gru_module_2(eta_d, alpha)
        self.m3 = gru_module_3()
        self.m4 = gru_module_4(n_agents, eta_c, eta_d)

    def forward(self, pi_g, pi_p, pi_s, pi_c, pi_d, d_t_n, C_p_n=None, C_d_n=None, C_g_n=None):
        '''
        Forward method of Neural GrU.

        Parameters
        ----------
        pi_g : torch tensor of shape (24, )
            Grid price
        
        pi_p : torch tensor of shape (24, )
            PV Energy prices

        pi_s : torch tensor of shape (24, )
            Storage prices

        pi_c : torch tensor of shape (24, )
            Charging prices

        pi_d : torch tensor of shape (24, )
            Discharging prices

        d_t_n : torch tensor of shape (n_agents, 24)
            Total demand vector of all agents

        C_p_n : torch tensor of shape (n_agents, 24), default=None
            Constraint of PV Energy demand for all agents
        
        C_d_n : torch tensor of shape (n_agents, 24), default=None
            Constraint of Discharging demand for all agents

        C_g_n : torch tensor of shape (n_agents, 24), default=None
            Constraint of Grid demand for all agents

        Returns
        -------
        d_star_n : torch tensor of shape (n_agetns, 5, 24)
            Demand breakup vectors of all agents in order [d_g, d_p, d_s, d_c, d_d]
        '''

        # ppi_p = F.relu(pi_p)
        # ppi_s = F.relu(pi_s)
        # ppi_c = F.relu(pi_c)
        # ppi_d = F.relu(pi_d)

        Pi_g_n = pi_g.expand(self.n_agents, -1) + self.ep_t_n
        Pi_p_n = F.relu(pi_p.expand(self.n_agents, -1))
        Pi_s_n = F.relu(pi_s.expand(self.n_agents, -1))
        Pi_c_n = F.relu(pi_c.expand(self.n_agents, -1))
        Pi_d_n = F.relu(pi_d.expand(self.n_agents, -1))

        if not torch.is_tensor(C_d_n):
            C_d_n = torch.ones((self.n_agents, 24), requires_grad=False, device=self.device) * self.alpha
        
        if not torch.is_tensor(C_g_n):
            C_g_n = torch.ones((self.n_agents, 24), requires_grad=False, device=self.device) * self.alpha

        if not torch.is_tensor(C_p_n):
            C_p_n = torch.ones((self.n_agents, 24), requires_grad=False, device=self.device) * self.alpha

        C_t_n = torch.stack([C_d_n.T * self.eta_d, C_p_n.T, C_g_n.T]).T

        pi_tilda_c_t, i_t = self.m1(Pi_s_n, Pi_c_n, Pi_g_n, self.B)
        A_t = self.m2(pi_tilda_c_t, Pi_d_n, Pi_p_n, Pi_g_n, self.B)
        i_tilda_t = self.m3(i_t)
        d_star_n = self.m4(d_t_n, i_t, A_t, i_tilda_t, C_t_n)

        return d_star_n