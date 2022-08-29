from enum import IntEnum

import numpy as np


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


def getCost(prices, demandBreakup):
    '''
    Compute the total cost to user for a given demand breakup & price vectors.

    Parameters
    ---------- 
    prices : ndarray of shape (5, T)
        Price vectors for g, p, s, c & d.

    demandBreakup : ndarray of shape (5, T)
        Demand vectors for g, p, s, c & d.

    Returns
    -------
    totalCost : float
        Total cost to user.
    '''
    cost = np.zeros(len(source))

    cost[source.charge] = np.dot( prices[source.charge], demandBreakup[source.charge] ) 
    cost[source.grid] = np.dot( prices[source.grid], demandBreakup[source.grid] )
    cost[source.pv] = np.dot( prices[source.pv], demandBreakup[source.pv] )
    cost[source.discharge] = np.dot( prices[source.discharge], demandBreakup[source.discharge] )
    cost[source.storage] = np.dot( prices[source.storage], demandBreakup[source.storage] )

    return np.sum( cost )


def getBestExporter( prices, t, eta_c, eta_d ):
    '''
    Get the best time slot & respective price to charge DER battery for later use.

    Parameters
    ---------- 
    prices : ndarray of shape (5, T)
        Price vectors for g, p, s, c & d.

    t : int
        Time slot to get the best exporter for.

    eta_c : int
        Efficiency constant for charging

    eta_d : int
        Efficiency constant for discharging

    Returns
    -------
    bestPrice : float
        Price corresponding to the best exporting time slot.

    bestExporter : int
        Best exporting time slot.
    '''
    buyAndChargePrice = prices[ source.grid, : ] + prices[ source.charge, : ] / ( eta_c * eta_d )
    cumStoragePrice = np.cumsum( prices[ source.storage, : ] / eta_d )

    # No possibility of import for the first time step
    if t == 0:
        return ( np.inf, -1 )
    # For the second time step, only the first time step can be used
    if t == 1:
        return ( prices[source.grid, 0] + (prices[source.charge, 0] / ( eta_c * eta_d )) + ( prices[ source.storage, 0 ] / eta_d ), 0)
    
    # Calculate export price for various starting points
    exportPrice = cumStoragePrice[ t-1 ] - np.append( 0, cumStoragePrice[ 0: t-1 ] )
    exportPrice += buyAndChargePrice[ 0: t ]
    
    return ( np.min( exportPrice ), np.argmin( exportPrice ) )


def GrU( demandTotal, prices, supply=None, eta_c=1, eta_d=1 ):
    '''
    GrU Algorithm calculates demand breakup & utility using Greedy approach.
    Utility = Cost to user w/o DER - Cost to user with DER

    Parameters
    ---------- 
    demandTotal : ndarray of shape (T, )
        Total demand vector.

    prices : ndarray of shape (5, T)
        Price vectors for g, p, s, c & d.

    supply : ndarray of shape (T, ), default = None
        PV supply vector. 

    eta_c : int, default = 1
        Efficiency constant for charging

    eta_d : int, default = 1
        Efficiency constant for discharging

    Returns
    -------
    demandBreakup : ndarray of shape (5, T, T)
        Demand vectors for g, p, s, c & d.

    util : float
        Utility for the demand breakup
    '''
    demandBreakup = np.zeros( prices.shape )

    if not supply:
        supply = np.zeros( demandTotal.shape )

    demand = demandTotal - supply
    demand[ demand < 0 ] = 0

    for t in range( prices.shape[1] ):
        ( bestPrice, bestExporter ) = getBestExporter( prices, t, eta_c, eta_d )
        
        # Battery discharge is the cheapest
        if bestPrice + prices[ source.discharge, t ] / eta_d < np.minimum( prices[ source.grid, t ], prices[ source.pv, t ] ):
            demandBreakup[ source.discharge, t ] = demand[ t ] / eta_d
            demandBreakup[ source.grid, t ] = 0
            demandBreakup[ source.pv, t ] = 0
            
            # Make arrangements to receive this power
            demandBreakup[ source.charge, bestExporter ] += demand[ t ] / ( eta_c * eta_d )
            demandBreakup[ source.grid, bestExporter ] += demand[ t ] / ( eta_c * eta_d )  
            for tau in range( bestExporter, t ):
                demandBreakup[ source.storage, tau ] += demand[ t ] / eta_d
                
        # Grid is the cheapest
        elif prices[ source.grid, t ] < prices[ source.pv, t ]:
            demandBreakup[ source.discharge, t ] = 0
            demandBreakup[ source.grid, t ] = demand[ t ]
            demandBreakup[ source.pv, t ] = 0
            
        # PV is the cheapest
        else:
            demandBreakup[ source.discharge, t ] = 0
            demandBreakup[ source.grid, t ] = 0
            demandBreakup[ source.pv, t ] = demand[ t ]

    # print(f"Cost without DER = {np.sum( demand * prices[source.grid] )}")
    # print(f"Cost wit DER = {getCost( prices, demandBreakup )}")

    # Utility = Cost to user without DER - Cost to user with DER
    util = np.sum( np.dot( demand, prices[source.grid] )) - getCost( prices, demandBreakup )
        
    return demandBreakup, util



def evaluate(true_d, pred_d):
    '''
    Computes the Mean Absolute Error & Mean Absolute Percentage Error in demand prediction w.r.t the true demands.

    Parameters
    ----------
    true_d : ndarray of shape (l ,T)
        True breakup of total demand in order p, s, c, d.

    pred_d : ndarray of shape (l, T)
        Predicted breakup of total demand in order p, s, c, d.

    Returns
    -------
    mae : array of shape (l, )
        MAE for predicted & true demands in order p, s, c, d.

    mape : array of shape (l, )
        MAPE for predicted & true demands in order p, s, c, d.
    '''

    l, T = true_d.shape

    mae = [0] * l
    for i in range(l):
        mae[i] = np.sum(np.abs(pred_d[i] - true_d[i]))
        mae[i] /= T

    mape = [0] * l
    for i in range(len(pred_d)):
        mape[i] = np.sum(np.abs(pred_d[i] - true_d[i]) / (true_d[i] + 1e-8))
        mape[i] /= T
    
    return mae, mape
    

if __name__ == '__main__':
    # Sample price and total demand vectors to test GrU
    prices = np.array( [ [ 0.3137888, 0.2407818, 0.1934758, 0.1749464, 0.165518, 0.1668898, 0.1911524, 0.2558408, 0.3410822, 0.3459716, 0.3265528, 0.309122, 0.3031012, 0.2993914, 0.307586, 0.2944496, 0.3057066, 0.3296806, 0.3864446, 0.4992488, 0.5209086, 0.5017106, 0.4622106, 0.3913898 ], [ 0.479, 0.467, 0.464, 0.463, 0.462, 0.462, 0.464, 0.467, 0.305, 0.313, 0.317, 0.290, 0.304, 0.291, 0.308, 0.295, 0.307, 0.471, 0.473, 0.487, 0.493, 0.491, 0.476, 0.476 ], [ 0.00050, 0.00050, 0.00120, 0.00280, 0.00525, 0.00770, 0.00945, 0.00945, 0.00915, 0.00900, 0.00885, 0.00885, 0.00880, 0.00880, 0.00880, 0.00925, 0.00995, 0.00845, 0.00545, 0.00275, 0.00050, 0.00050, 0.00050, 0.00050 ], [ 0.010, 0.010, 0.033, 0.053, 0.066, 0.070, 0.052, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.024, 0.022, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010 ], [ 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.080, 0.081, 0.080, 0.010, 0.010 ] ] )
    demandTotal = np.array( [ 6.977, 4.481, 5.309, 3.792, 3.726, 3.611, 4.774, 6.38, 8.807, 7.441, 6.983, 6.836, 7.285, 7.132, 7.743, 7.396, 9.289, 8.745, 9.847, 12.738, 15.76, 16.945, 15.414, 11.289 ] )

    breakup, util = GrU( demandTotal, prices, eta_c = 0.9487, eta_d = 0.9487 )

    print( "Demand sums up properly: ", np.allclose( breakup[ source.discharge, : ]*0.9487 + breakup[ source.grid, : ] + breakup[ source.pv, : ], demandTotal ) )
    consumptionSources = [ source.discharge, source.grid, source.pv ]
    breakupConsumption = breakup[ consumptionSources, : ]
    maskConsumption = np.zeros( breakupConsumption.shape )
    np.place( maskConsumption, breakupConsumption > 0 , 1 )
    print( "Exactly one source is consumed at any time: ", np.all( np.sum( maskConsumption, 0 ) == 1 ) )

    CDSources = [ source.charge, source.discharge ]
    breakupCD = breakup[ CDSources, : ]
    maskCD = np.zeros( breakupCD.shape )
    np.place( maskCD, breakupCD > 0 , 1 )
    print( "Either charge or discharge at any time: ", np.all( np.sum( maskCD, 0 ) <= 1 ) )

    print( breakup )
