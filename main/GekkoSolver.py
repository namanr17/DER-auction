from enum import IntEnum

import numpy as np

from gekko import GEKKO
from GrU import getCost


class source( IntEnum ):
    grid =  0
    pv = 1
    storage = 2
    charge = 3
    discharge = 4

# For clock data
class clock( IntEnum ):
    pv = 0
    storage = 1
    charge = 2
    discharge = 3

def gekkoDemandSolver(demand, prices, supply=None, eta_c=1, eta_d=1, ReLU=True):
    '''
    **Gekko Solver** without Env. Impact.
    Modified bid() method from main_auction.py

    - Added ReLU to handle negative demands.
    - Returns Utility (Cost w/o DER - Cost with DER) with the generated demand breakup.
    - Grid demand is assumed to be $ d_g =  d_{total} - d_p - d_d $ (to be updated).
    
    Parameters
    ---------- 
    demand : ndarray of shape (T, )
        Total demand vector.

    prices : ndarray of shape (5, T)
        Price vectors for g, p, s, c & d.

    supply : ndarray of shape (T, ), default = None
        PV supply vector. 

    eta_c : int, default = 1
        Efficiency constant for charging

    eta_d : int, default = 1
        Efficiency constant for discharging

    ReLU : bool, default = True
        Whether to apply the Rectifier or ReLU Activation function on the demand vectors.

    Returns
    -------
    demandBreakup : ndarray of shape (5, T, T)
        Demand vectors for g, p, s, c & d.

    util : float
        Utility for the demand breakup


    '''
    m = GEKKO(remote=False)
    t = 24

    grid_price = prices[source.grid]
    price_capacity = prices[source.storage]
    price_energy = prices[source.pv]
    price_charging = prices[source.charge]
    price_discharging = prices[source.discharge]

    if not supply:
        supply = np.zeros( demand.shape )

    a = [m.Var() for n in range(t)]
    e_pv = [m.Var(lb=0) for n in range(t)]

    def cost_wo():
        cost = 0
        for i in range(t):
            load = demand[i] - supply[i]
            if load >= 0:
                cost = cost + load * grid_price[i]
            else:
                cost = cost
        return cost

    def soc(i):
        soc_a = 0
        for t in range(i+1):
            soc_a += a[t]
        return soc_a

    def cost_w():
        cost = 0
        for i in range(t):
            load = demand[i] - supply[i] - e_pv[i]\
                   + 0.5*(m.tanh(100*a[i])+1) * a[i]/eta_c - 0.5*(m.tanh(100*a[i])-1) * a[i]*eta_d
            cost += load * grid_price[i] + soc(i) * price_capacity[i] + e_pv[i] * price_energy[i]\
                    + 0.5* (m.tanh(100*a[i])+1) * a[i] * price_charging[i] \
                    + 0.5* (m.tanh(100*a[i])-1) * a[i] * price_discharging[i]
        return cost

    utility = cost_wo() - cost_w()

    for i in range(t):
        load = demand[i] - supply[i] - e_pv[i] + a[i]
        m.Equation(load >= 0)

    for i in range(t):
        m.Equation(soc(i) >= 0)

    m.Equation(soc(t-1) == 0)
    m.options.IMODE = 3
    m.options.MAX_ITER = 1500
    m.Maximize(utility)

    try:
        m.solve(disp=False)
        max_utility = -1 * m.options.OBJFCNVAL
        a_optimal = np.zeros(t)
        e_optimal = np.zeros(t)
        for i in range(t):
            a_optimal[i] = a[i][0]
            e_optimal[i] = e_pv[i][0]
    except:
        print('no solution')
        max_utility = 0
        a_optimal = np.zeros(t)
        e_optimal = np.zeros(t)


    m.cleanup()
    def soc_optimal(i):
        soc_a = 0
        for t in range(i+1):
            soc_a += a_optimal[t]
        return soc_a

    demand_capacity = np.zeros(t)
    demand_charging = np.zeros(t)
    demand_discharging = np.zeros(t)

    for i in range(t):
        demand_capacity[i] = soc_optimal(i)
        if a_optimal[i] >= 0:
            demand_charging[i] = a_optimal[i]
        else:
            demand_discharging[i] = -1 * a_optimal[i]

    demandBreakup = np.ndarray( prices.shape )
    
    demandBreakup[ source.pv ] = e_optimal
    demandBreakup[ source.charge ] = demand_charging
    demandBreakup[ source.discharge ] = demand_discharging
    demandBreakup[ source.storage ] = demand_capacity
    # demandBreakup[ source.grid ] = demand - e_optimal - demand_discharging
    demandBreakup[ source.grid ] = demand - e_optimal - demand_discharging + demand_charging

    if ReLU:
        demandBreakup[demandBreakup < 0] = 0

    util = np.sum( np.dot( demand, prices[source.grid] )) - getCost( prices, demandBreakup )

    return demandBreakup, util
