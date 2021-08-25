"""
This modules efficiently tests 2D points to see
if they are located within or outside of
a polygon.

Taken/adapted from (docstrings added and minor enhancements):
REF: sasamil, March 2021 (commit: a31d541)
https://github.com/sasamil/PointInPolygon_Py/blob/master/pointInside.py

Numba (jit) additions, is_inside_sm_parallel() and benchmarks:
REF: Mehdi, 2013
https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
"""

import numpy as np
import numba
from numba import jit, njit
from typing import Sequence


@jit(nopython=True)
def is_inside_sm(polygon, point):
    """
    This function gives the answer whether the given point is inside or outside the predefined polygon
    Unlike standard ray-casting algorithm, this one works on edges! (with no performance cost)
    According to performance tests - this is the best variant.
    Attention! The [polygon length] list itself has an additional member - the last point coincides with the first.

    Parameters
    ----------
    polygon : NumPy array
        Searched polygon. Last point should coincide with the first.

    point : NumPy array
        An arbitrary point that can be inside or outside the polygon.


    Returns
    -------
    int
        0 - the point is outside the polygon.
        1 - the point is inside the polygon.
        2 - the point is one edge (boundary).

    
    Example
    -------
    ```
    lenpoly = 100
    polygon = [[np.sin(x)+0.5,np.cos(x)+0.5]
        for x in np.linspace(0,2*np.pi,lenpoly)]
    polygon = np.array(polygon)
    N = 1000000
    points = np.random.uniform(-1.5, 1.5, size=(N, 2))
    is_inside_sm_parallel(points,polygon)
    ```
    """

    # Ignore final member? (should be same as first)
    length = len(polygon)-1
    dy2 = point[1] - polygon[0][1]
    intersections = 0
    ii = 0
    jj = 1

    while ii<length:
        dy  = dy2
        dy2 = point[1] - polygon[jj][1]

        # consider only lines which are not completely above/bellow/right from the point
        if dy*dy2 <= 0.0 and (point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]):
            
            # non-horizontal line
            if dy<0 or dy2<0:
                F = dy*(polygon[jj][0] - polygon[ii][0])/(dy-dy2) + polygon[ii][0]

                if point[0] > F: # if line is left from the point - the ray moving towards left, will intersect it
                    intersections += 1
                elif point[0] == F: # point on line
                    return 2

        # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
        elif dy2==0 and (point[0]==polygon[jj][0] or (dy==0 and (point[0]-polygon[ii][0])*(point[0]-polygon[jj][0])<=0)):
            return 2

        # there is another posibility: (dy=0 and dy2>0) or (dy>0 and dy2=0). It is skipped 
        # deliberately to prevent break-points intersections to be counted twice.
        
        ii = jj
        jj += 1
                
    #print 'intersections =', intersections
    return intersections & 1


@njit(parallel=True)
def is_inside_sm_parallel(polygon, points):
    """
    Numba parallel version of is_inside_sm() for multiple points.
    This function gives the answer whether the given point is inside or outside the predefined polygon
    Unlike standard ray-casting algorithm, this one works on edges! (with no performance cost)
    According to performance tests - this is the best variant.
    Attention! The [polygon] list itself has an additional member - the last point coincides with the first.

    Parameters
    ----------
    polygon : NumPy array
        Searched polygon. Last point should coincide with the first.

    points : NumPy array
        A array of arbitrary points inside or outside the polygon.

    return_bool : bool
        See Returns.

    Returns
    -------
    NumPy array(boolean)
        False - the point is outside the polygon.
        True - the point is inside the polygon or on the edge.

    Example
    -------
    ```
    lenpoly = 100
    polygon = [[np.sin(x)+0.5,np.cos(x)+0.5]
        for x in np.linspace(0,2*np.pi,lenpoly)]
    polygon = np.array(polygon)
    N = 1000000
    points = np.random.uniform(-1.5, 1.5, size=(N, 2))
    is_inside_sm_parallel(points,polygon)
    ```
    """

    ln = len(points)
    D = np.empty(ln, dtype=numba.boolean) 
    for i in numba.prange(ln):
        D[i] = is_inside_sm(polygon,points[i])
    return D