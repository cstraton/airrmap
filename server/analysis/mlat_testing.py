# Testing multilateration (figure, circle distances)
# REF: Adapted from Alan Zucconi, March 2017 https://www.alanzucconi.com/2017/03/13/positioning-and-trilateration/
# REF: Great circle distance: https://www.alanzucconi.com/2017/03/13/understanding-geographical-coordinates/

# USE
# 1. Set CONFIG.
# 2. Will write out image to current folder.

# %% imports
import math
import random
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.optimize import OptimizeResult
import plotly.express as px
import plotly.graph_objects as go
from plotly import plot


# %% CONFIG
CLOSEST_N = 5  # Closest anchors
ERROR_FRACTION = 0.2
OUT_NAME = f'mlat_{CLOSEST_N}_anchors_{ERROR_FRACTION}_error.png'

# %% Calculate Euclidean distance


def calc_distance(x1, y1, x2, y2):
    """Calculate the Euclidean distance
    between a pair of Cartesian coordinates
    (c^2 = a^2 + b^2)"""

    dx = (x2 - x1)**2
    dy = (y2 - y1)**2
    return (dx + dy)**0.5


# %% Mean Squared Error
def mse(x, locations, distances):
    # NOTE: Note call to calc_distance
    # REF: https://www.alanzucconi.com/2017/03/13/positioning-and-trilateration/
    # Mean Square Error
    # locations: [ (lat1, long1), ... ]
    # distances: [ distance1, ... ]
    mse = 0.0
    for location, distance in zip(locations, distances):
        distance_calculated = calc_distance(
            x[0], x[1], location[0], location[1])
        mse += math.pow(distance_calculated - distance, 2.0)
    return mse / len(distances)


# %% Function to add a new point record (for plotting)
def append_point(df, id_ref, x, y, category):
    """Add a new point row to the dataframe"""
    df_point = pd.DataFrame()
    df_point = df_point.append(pd.Series(name=id_ref))
    df_point['id'] = id_ref
    df_point[['x', 'y']] = x, y
    df_point['category'] = category
    df_point[['dx', 'dy', 'c2']] = [0, 0, 0]
    df_point['dist'] = 2  # used for size
    return df.append(df_point)


# %% Define random point
# (just for testing purposes)
target_xy = ((2, 3.5))
df_plot = pd.DataFrame()
# df_plot = append_point(df_plot, 'target_start',
#                      target_xy[0], target_xy[1], 'target_start')
point_x = np.array(target_xy)  # Define point in space for testing only


# %% Define known locations
anchor_coords = [
    {'id': 'a', 'x': 1, 'y': 2},
    {'id': 'b', 'x': 2, 'y': 4},
    {'id': 'c', 'x': 1, 'y': 1},
    {'id': 'd', 'x': 3, 'y': 2},
    {'id': 'e', 'x': 4, 'y': 3},
    {'id': 'f', 'x': 3.5, 'y': 4},
    {'id': 'g', 'x': 3, 'y': 0},
    {'id': 'h', 'x': 0, 'y': 0.5}
]

df_anchors = pd.DataFrame(anchor_coords)
df_anchors['category'] = 'anchor'
df_anchors.index = list(df_anchors['id'])


# %% Calculate distance (x,y) to target
df_anchors[['dx', 'dy']] = point_x - df_anchors[['x', 'y']]

# %% Calculate distance
df_anchors['c2'] = df_anchors['dx']**2 + df_anchors['dy']**2
df_anchors['dist'] = df_anchors['c2']**0.5

# %% Add some error to the distance
df_anchors['dist'] = df_anchors['dist'] * \
    np.random.uniform(
        1. - ERROR_FRACTION,
        1. + ERROR_FRACTION,
    len(df_anchors)
)


# %% Get the x closest anchors (smallest distances)
# np.argpartition will return full list unsorted
# but the closest N records will be the first N records
# in the list
ind_closest = np.array(np.argpartition(
    df_anchors['dist'], CLOSEST_N))  # loses coords
# only first few are the closest (returns full unsorted list though)
df_anchors_closest = df_anchors.iloc[ind_closest[:CLOSEST_N]]

# %% Update category for closest anchors
df_anchors['category'] = \
    df_anchors.apply(
        lambda row: 'anchor_closest' if row['id'] in list(
            df_anchors_closest['id']) else row['category'],
        axis='columns'
)


# %% CHECK
# Add midpoint
anchor_midpoint_coord = df_anchors_closest[['x', 'y']].mean()
df_plot = append_point(
    df_plot,
    'midpoint_start',
    anchor_midpoint_coord[0],
    anchor_midpoint_coord[1],
    'midpoint_start'
)


# %% ==== Compute the location ====

# locations: [ (lat1, long1), ... ]
# distances: [distance1, ....]

# anchord_coords_
# minimize

initial_location = anchor_midpoint_coord
locations = df_anchors_closest[['x', 'y']].values
distances = df_anchors_closest[['dist']].values
minimize_history = []


# %% Define callback
def callback_minimize(xk, state: OptimizeResult, history: list):
    global minimize_history
    minimize_history.append((state))


# %% Minimize the distance
# # REF: https://www.alanzucconi.com/2017/03/13/positioning-and-trilateration/
result = minimize(
    mse,                    # Mean Square Error function
    initial_location,       # Initial guess
    args=(locations, distances),  # Additional parameters for mse
    method='L-BFGS-B',      # The optimisation algorithm
    # callback=callback_minimize, # callback function for history
    options={
        'ftol': 1e-5,        # Tolerance
        'maxiter': 1e+7     # Maximum iterations
    }
)

hist_values = [y.x for y in minimize_history]
print(hist_values)
# print(result.x)


# Orig
# options={
#       'ftol': 1e+7,        # Tolerance
#       'maxiter': 1e+7     # Maximum iterations
#   }


# %% Create plot DataFrame
df_plot = df_plot.append(df_anchors[:])
df_plot = append_point(
    df=df_plot,
    id_ref='target_final',
    x=result.x[0],
    y=result.x[1],
    category='target_final'
)


# %% Add fixed point_size
# (required for plotly)
df_plot['anchor_size'] = 1

# Define colours
color_discrete_map = dict(
    # midpoint_start='#009988',  # Teal
    midpoint_start='#117733',
    # anchor_closest='#33BBEE',  # Cyan
    anchor_closest='#000000',  # Black
    anchor='rgba(0,0,0,0)',  # Hide (transparent)
    target_final='#CC3311'  # Red
)

# %% Draw figure (anchors and points)
#  REF: https://stackoverflow.com/questions/53217404/specifying-marker-size-in-data-unit-for-plotly

fig = px.scatter(
    df_plot,
    x='x',
    y='y',
    size='anchor_size',
    color='category',
    hover_data=['dist', 'id'],
    # color_discrete_sequence=px.colors.qualitative.Vivid
    color_discrete_map=color_discrete_map,
    opacity=0.9
)

# Add dashed circles
kwargs = {'type': 'circle',
          'xref': 'x',
          'yref': 'y',
          'fillcolor': None,
          'line_color': color_discrete_map['anchor_closest'],
          'line_width': 2,
          'opacity': 0.4,
          'line_dash': 'dot'
          }
points = [go.layout.Shape(x0=x - r, y0=y - r, x1=x + r, y1=y + r, **kwargs)
          for x, y, r in df_anchors_closest[['x', 'y', 'dist']].values]

# Add target circle
r = 0.3
x, y = target_xy
target_kwargs = kwargs.copy()
target_kwargs['line_dash'] = 'solid'
target_kwargs['line_width'] = 3
target_kwargs['line_color'] = color_discrete_map['target_final']
target_kwargs['opacity'] = 1.0

points.append(go.layout.Shape(x0=x - r, y0=y - r,
              x1=x + r, y1=y + r, **target_kwargs))


# Arrow (initial position to final)
arrow = go.layout.Annotation(dict(
    x=result.x[0],
    y=result.x[1],
    xref='x',
    yref='y',
    showarrow=True,
    axref='x',
    ayref='y',
    ax=initial_location[0],
    ay=initial_location[1],
    arrowhead=2,
    arrowwidth=2.5,
    arrowcolor='black')
)

# Force same aspect ratio
fig.update_xaxes(
    range=[-1, 6.5],
    visible=True,
    tick0=0,
    dtick=2
)
fig.update_yaxes(
    scaleanchor="x",
    scaleratio=1,
    range=[-1, 6.5],
    visible=True,
    tick0=0,
    dtick=2
)

# Set marker size
fig.update_traces(marker=dict(
    size=27,
    line=dict(
        width=3,
        color='white'
    )
))

fig.update_layout(
    shapes=points,
    annotations=[arrow],
    width=600,
    height=600,
    showlegend=False,
    template='presentation')
fig.update_traces(textposition='middle right')
fig.show()
fig.write_image(f'results/{OUT_NAME}')
