# Obtain spatial coordinates from anchor distances
# Adapted from work by Alan Zucconi:
# REF: Positioning and trilateration, March 2017 https://www.alanzucconi.com/2017/03/13/positioning-and-trilateration/
# REF: Great circle distance: https://www.alanzucconi.com/2017/03/13/understanding-geographical-coordinates/

# %% imports
import math
from numba.core.decorators import jit, njit
from scipy.optimize import minimize

# Define calc_distance
@njit
def calculate_distance(x1, y1, x2, y2):
    """Calculate the Euclidean distance
    between a pair of Cartesian coordinates
    (c^2 = a^2 + b^2)"""

    # This is related to 
    # the plot distance (not AIR distance)

    dx = (x2 - x1)**2
    dy = (y2 - y1)**2
    return (dx + dy)**0.5



# %% Mean Squared Error
# NOTE: Note call to calc_distance
# REF: https://www.alanzucconi.com/2017/03/13/positioning-and-trilateration/
# Mean Square Error
# locations: [ (lat1, long1), ... ] # TODO: Check if long1, lat1
# distances: [ distance1, ... ]
@njit
def mse(x, locations, distances):
    mse = 0.0
    for location, distance in zip(locations, distances):
        distance_calculated = calculate_distance(
            x[0], x[1], location[0], location[1])
        mse += math.pow(distance_calculated - distance, 2.0)
    return mse / len(distances)


@njit
def mae(x, locations, distances):
    """Compute the mean absolute error"""
    mae = 0.0
    for location, distance in zip(locations, distances):
        distance_calculated = calculate_distance(
            x[0], x[1], location[0], location[1])
        mae += math.fabs(distance_calculated - distance)
    return mae / len(distances)


def calculate_coords(initial_coords, anchor_coords, anchor_distances, use_mae=False):
    """Calculate xy coordinates using distances to anchors

    NOTE!: Ensure to pass anchor_coords and anchor_distances as ndarray
           and not list/tuple, ~78x faster in testing.

    TODO: Complete docstring description (types)

    Args:
        initial_coords (Sequence[int,int]): Initial guess
        anchor_coords (ndarray): Array of anchor coordinates [[x, y]]
        anchor_distances (ndarray): Array of anchor distances [x]
        use_mae (bool): True to use mean absolute error instead of mean squared error.

    Returns:
        res (OptimizeResult): e.g. x, y = result.x. Check res.success is True.
            res.message will contain the error if res.success if False.
    """

    # mse or mae
    obj_func = mae if use_mae else mse

    # %% Minimize the distance
    # # REF: https://www.alanzucconi.com/2017/03/13/positioning-and-trilateration/
    result = minimize(
        obj_func,                    # Mean Square Error function
        initial_coords,         # Initial guess
        args=(anchor_coords, anchor_distances),  # Additional parameters for mse
        method='L-BFGS-B',      # The optimisation algorithm
        # callback=callback_minimize, # callback function for history
        options={
            'ftol': 1e-5,        # Tolerance
            'maxiter': 1e+7     # Maximum iterations
        }
    )

    return result
