import numpy as np
import pandas as pd
from gekko import GEKKO
from time import time
start = time()


###  initialisation  ###

DF = pd.read_csv('./data/community_data_25_17oct.csv')

grid_price = DF['GRID_PRICE']
emf = DF['EMF_17oct15']

energy_pv = DF['PV_DER']
BESS_capacity = 54
P_charging = 14.72
P_discharging = 14.72
eta_c = 0.9487
eta_d = 0.9487

customers = 25
t = 24

###  clock auction  ###


# price adjustment

delta_capacity = 0.00005
delta_charging = 0.001
delta_discharging = 0.001
delta_energy = 0.001



## utilities and bidding in the clock auction

weights = pd.read_csv('./data/weights.csv')
weights_env = weights['weights_env']


def bid(agent):
    m = GEKKO(remote=False)
    t = 24
    identity_demand = 'D' + str(agent)
    identity_generation = 'PV' + str(agent)
    demand = DF[identity_demand]
    supply = DF[identity_generation]

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

    def impact_wo():
        impact = 0
        for i in range(t):
            load = demand[i] - supply[i]
            if load >= 0:
                impact += load * emf[i]
            else:
                impact = impact
        return impact

    def impact_w():
        impact = 0
        for i in range(t):
            load = demand[i] - supply[i] - e_pv[i]\
                   + 0.5 * (m.tanh(100 * a[i]) + 1) * a[i] / eta_c - 0.5 * (m.tanh(100 * a[i]) - 1) * a[i] * eta_d
            impact += load * emf[i]
        return impact

    utility = cost_wo() - cost_w() + weights_env[agent] * (impact_wo() - impact_w())

    for i in range(t):
        load = demand[i] - supply[i] - e_pv[i] + a[i]
                   #+ 0.5 * (m.tanh(100 * a[i]) + 1) * a[i] / eta_c - 0.5 * (m.tanh(100 * a[i]) - 1) * a[i] * eta_d
        m.Equation(load >= 0)
        # m.Equation(a[i] <= 0.1*P_charging)
        # m.Equation(a[i] >= -0.1*P_discharging)

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

    bid_demand = {'demand_capacity': demand_capacity, 'demand_charging': demand_charging,
                    'demand_discharging': demand_discharging, 'demand_energy': e_optimal}
    print("Agent: ", agent+1)

    print("utility:", max_utility)
    return bid_demand


# main auction
clock_dataset = {}
counter = 0
while counter < 10000:
    print("###    clock round", str(counter), "    ###")
    time_round = time()
    counter += 1
    BESS_prices = pd.read_csv('./prices.csv')
    price_capacity = BESS_prices['capacity']
    price_charging = BESS_prices['charging']
    price_discharging = BESS_prices['discharging']
    price_energy = BESS_prices['energy']

    price_identifier = 'price_' + str(counter-1)
    price_round = []
    price_round.append(price_energy)
    price_round.append(price_capacity)
    price_round.append(price_charging)
    price_round.append(price_discharging)
    clock_dataset[price_identifier] = price_round
    # get demand (bid)
    bids_capacity = {}
    bids_charging = {}
    bids_discharging = {}
    bids_energy = {}
    for i in range(customers):
        demand = bid(i)
        demand_capacity = demand['demand_capacity']
        demand_charging = demand['demand_charging']
        demand_discharging = demand['demand_discharging']
        demand_energy = demand['demand_energy']
        identifier = 'agent' + str(i+1)
        bids_capacity[identifier] = demand_capacity
        bids_charging[identifier] = demand_charging
        bids_discharging[identifier] = demand_discharging
        bids_energy[identifier] = demand_energy
        agentround_identifier = 'agent' + str(i+1) + str(counter-1)
        agent_demand = []
        agent_demand.append(demand_energy)
        agent_demand.append(demand_capacity)
        agent_demand.append(demand_charging)
        agent_demand.append(demand_discharging)
        clock_dataset[agentround_identifier] = agent_demand

    df1 = pd.DataFrame(bids_capacity)
    df1['total'] = df1.sum(axis=1)
    df1.to_csv('capacity_bids.csv', index=False)
    total_capacity = df1['total']

    df2 = pd.DataFrame(bids_charging)
    df2['total'] = df2.sum(axis=1)
    df2.to_csv('charging_bids.csv', index=False)
    total_charging = df2['total']

    df3 = pd.DataFrame(bids_discharging)
    df3['total'] = df3.sum(axis=1)
    df3.to_csv('discharging_bids.csv', index=False)
    total_discharging = df3['total']

    df4 = pd.DataFrame(bids_energy)
    df4['total'] = df4.sum(axis=1)
    df4.to_csv('energy_bids.csv', index=False)
    total_energy = df4['total']

    # checking if demand < supply
    condition_c = 0
    condition_pc = 0
    condition_pd = 0
    condition_e = 0
    for i in range(t):
        if total_capacity[i] > BESS_capacity:
            condition_c += 1
            price_capacity[i] += delta_capacity
        if total_charging[i] > P_charging:
            condition_pc += 1
            price_charging[i] += delta_charging
        if total_discharging[i] > P_discharging:
            condition_pd += 1
            price_discharging[i] += delta_discharging
        if total_energy[i] > energy_pv[i]:
            condition_e += 1
            price_energy[i] += delta_energy


    price = {'capacity': price_capacity, 'charging': price_charging,
             'discharging': price_discharging, 'energy': price_energy}
    df5 = pd.DataFrame(price)
    df5.to_csv('prices.csv', index=False)

    conditions = condition_c + condition_pc + condition_pd + condition_e
    print("#######     conditions violated:", conditions, "    #######")
    print(f'round time: {time() - time_round} seconds')
    if condition_c == 0:
        if condition_pc == 0:
            if condition_pd == 0:
                if condition_e == 0:
                    break

df6 = pd.DataFrame(clock_dataset)
df6.to_csv('clock_data.csv', index=False)
print(f'Time taken to run: {time() - start} seconds')

