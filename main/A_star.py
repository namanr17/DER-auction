from enum import IntEnum

import numpy as np

from GrU import getBestExporter


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


def getAStarMatrices (prices, T=24, eta_c=1, eta_d=1):
    '''
    Computes A* matrices for g, p, s, c & d from the price vectors.
    Since it is based on GrU Algorithm, matrices are filled in similar fashion.

    Parameters
    ----------
    prices : ndarray of shape (5, T)
        Price vectors for g, p, s, c & d.

    T : int, default = 24
        Number of time slots (hours) to compute A* matrices for

    eta_c : int, default = 1
        Efficiency constant for charging

    eta_d : int, default = 1
        Efficiency constant for discharging

    Returns
    -------
    A_star : ndarray of shape (5, T, T)
        A* matrices for g, p, s, c & d.
    '''
    A_star = np.zeros(shape=( len(source), T, T ))

    # Fill A* matrices
    for t in range(T):
        ( bestPrice, bestExporter ) = getBestExporter( prices, t, eta_c, eta_d )

        # Battery discharge is the cheapest
        if bestPrice + prices[ source.discharge, t ] / eta_d < np.minimum( prices[ source.grid, t ], prices[ source.pv, t ] ):
            A_star[ source.discharge, t, t ] = 1 / eta_d
            A_star[ source.charge, bestExporter, t ] += 1 / (eta_c * eta_d)
            A_star[ source.grid, bestExporter, t ] += 1 / (eta_c * eta_d)
            
            for tau in range( bestExporter, t-1 ):
                A_star[ source.storage, tau, t ] += 1 / eta_d
            
        # Grid is the cheapest
        elif prices[ source.grid, t ] < prices[ source.pv, t ]:
            A_star[ source.grid, t , t] = 1

        # PV is the cheapest
        else:
            A_star[ source.pv, t, t ] = 1
            
    return A_star


def estimateTotalDemandUsingAStar( prices, d_star, T=24, ReLU=True, alpha=1e-6, eta_c = 1, eta_d = 1 ):
    '''
    Estimates total demand vector using A* matrices and demand vectors for p, c, d & s.
    Uses the (Moore-Penrose) pseudo-inverse of a matrix to solve the least-squares problem.

    Parameters
    ----------
    prices : ndarray of shape (5, T, T)
        Price vectors for g, p, s, c & d.

    d_star : ndarray of shape (4, T)
        Demand vectors for p, s, c & d.

    T : int, default = 24
        Number of time slots (hours) to estimate the total demand vector for.

    ReLU : bool, default = True
        Whether to apply the Rectifier or ReLU Activation function on the estimated total demand vector.

    alpha : float, default = 1e-6
        Laplacian regularization factor for pseudo-inverse.

    eta_c : int, default = 1
        Efficiency constant for charging

    eta_d : int, default = 1
        Efficiency constant for discharging

    Returns
    -------
    d_pred : array of shape (24,)
        Estimated total demand vector.
    '''

    A_star = getAStarMatrices( prices, T, eta_c, eta_d )[1:]

    left_term = np.zeros( shape=( T, T ), dtype=float)
    right_term = np.zeros( shape=( T ), dtype=float)

    for e in clock:
        left_term += np.matmul( A_star[e].T, A_star[e] )
        right_term += np.matmul( A_star[e].T, d_star[e] )

    d_pred = np.dot( np.linalg.pinv( left_term + alpha * np.eye(T)), right_term )

    if ReLU:
        d_pred[d_pred < 0] = 0

    return d_pred
