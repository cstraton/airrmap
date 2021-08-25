# Functionality for loading anchors for the application.

# %% Imports
import pandas as pd
import json
import sqlite3
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# %%


def get_anchors_json(fn_anchors: str,
                     scaler_xy: MinMaxScaler,
                     v_group_le: LabelEncoder,
                     class_rgb) -> str:
    """Return the list of anchors in json format

    Args:
        fn_anchors (str): Database filename containing the anchors.

        scaler_xy (MinMaxScaler): Pre-fitted scaler for coordinates to tile space.

        v_group_le (LabelEncoder): Pre-fitted label encoder for V group.

        class_rgb (ndarray[tuple]): Numpy array of RGB tuples for each class encoded
            for by v_group_le.

    Returns:
        str: Json list of anchor records.
    """

    sql = 'select anchor_id, anchor_name, x, y from anchor_coords'

    with sqlite3.connect(fn_anchors) as conn:
        df: pd.DataFrame = pd.read_sql(sql, conn, index_col=None)

    #df[['x', 'y']] = xy_scaler.transform(df[['x', 'y']])

    # +ve y values are up
    # scale to required range
    df['x'] = scaler_xy.transform(df[['x']])
    df['y'] = scaler_xy.transform(df[['y']])
    df['v_group'] = df['anchor_name'].map(lambda x: x[:5]) # TODO: Make configurable.
    df['class_index'] = v_group_le.transform(df['v_group'])
    df['class_rgb'] = df['class_index'].map(lambda x: class_rgb[x])

    # reverse y, so higher y values are down.
    #df['y'] = df['y'].max() - df['y']

    return df.to_json(orient='records')
