# Anchor processing

# Imports
import pandas as pd
import sqlite3
# import umap (performed further down)
import plotly.express as px
from plotly.graph_objs._figure import Figure
from sklearn import manifold
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform



def _coord_distances(df_coords: pd.DataFrame, df_orig_dist: pd.DataFrame) -> pd.DataFrame:
    """Create square matrix of plotted distances

    Args:
        df_coords (DataFrame): Computed xy coordinates for each sequence.

        df_orig_dist (DataFrame): The original distance matrix (just used to
            get the list of row and column headers).

    Returns:
        pd.DataFrame: A square matrix with Euclidean distances between
            plotted points.
    """

    # Euclidean distance between plotted points.
    # Condensed distance matrix.
    plot_dist_matrix = pdist(df_coords[['x', 'y']])

    # Condensed to Square + add headers
    plot_dist_matrix = squareform(plot_dist_matrix)
    assert (plot_dist_matrix.shape == df_orig_dist.shape)

    # DataFrame with index and column headers
    df_plot_dist = pd.DataFrame(plot_dist_matrix,
                                index=list(df_orig_dist.index),
                                columns=list(df_orig_dist.columns))
    assert (df_plot_dist.shape == df_orig_dist.shape)

    # Return
    return df_plot_dist


def _coord_melt_distances(df_orig_dist: pd.DataFrame,
                          df_plot_dist: pd.DataFrame,
                          df_dist_delta: pd.DataFrame) -> pd.DataFrame:
    """Create long table of original vs plotted distances

    Args:
        df_orig_dist (pd.DataFrame): Original pairwise computed distances
        df_plot_dist (pd.DataFrame): Plotted pairwise distances
        df_dist_delta (pd.DataFrame): Difference between original and plotted distances

    Returns:
        pd.DataFrame: Long DataFrame combining data from all DataFrames
    """

    # Convert to tall tables
    df_orig_dist_melt = df_orig_dist.reset_index().melt(id_vars='index',
                                                        var_name='item2', value_name='orig_distance')
    df_orig_dist_melt.set_index(['index', 'item2'], inplace=True)

    df_plot_dist_melt = df_plot_dist.reset_index().melt(id_vars='index',
                                                        var_name='item2', value_name='plot_distance')
    df_plot_dist_melt.set_index(['index', 'item2'], inplace=True)

    df_dist_delta_melt = df_dist_delta.reset_index().melt(id_vars='index',
                                                          var_name='item2', value_name='dist_delta')
    df_dist_delta_melt.set_index(['index', 'item2'], inplace=True)

    # Join all three by item1 and item2 indexes
    df_melted = df_orig_dist_melt.join(
        df_plot_dist_melt, how='left', sort=True)
    df_melted = df_melted.join(df_dist_delta_melt, how='left', sort=True)
    df_melted.rename(columns={'index': 'item1'},
                     inplace=True)  # rename index -> item1

    # Return
    return df_melted


def compute_coords(df_dist_matrix: pd.DataFrame,
                   method: str,
                   random_state: int,
                   n_neighbors: int = 15):
    """Computes anchor coordinates from distance matrix

    Args:
        df_dist_matrix (DataFrame): Square distance matrix.

        method (str): Dimensionality reduction method. Valid values are
            'MDS', 'TSNE', 'PCA' or 'UMAP'.

        random_state (int): Random state (int to fix, None for random).

        n_neighbors (int): UMAP only, number of closest neighbours.

    Raises:
        Exception: If invalid value is supplied for method.

    Returns:
        Tuple[DataFrame, DataFrame]: Coordinates [0] and DataFrame 
            with original distances, plotted distances and difference between the two
    """

    # Fit and compute coordinates
    coords = None
    if method == 'PCA':
        model = PCA(n_components=2, random_state=random_state)
        coords = model.fit_transform(df_dist_matrix)
    elif method == 'TSNE':
        model = manifold.TSNE(n_components=2, random_state=random_state)
        coords = model.fit_transform(df_dist_matrix)
    elif method == 'MDS':
        model = manifold.MDS(n_components=2,
                             dissimilarity="precomputed",
                             random_state=random_state)
        coords = model.fit(df_dist_matrix).embedding_  # [[x,y]] coordinates
    elif method == 'UMAP':
        import umap  # Potentially slow to import due to Numba compile?
        model = umap.UMAP(n_neighbors=n_neighbors,
                          n_components=2,
                          random_state=random_state)
        coords = model.fit_transform(df_dist_matrix)

    else:
        raise Exception(f"Invalid method: '{method}'")

    # Dataframe of coordinates
    df_coords = pd.DataFrame(coords,
                             index=list(df_dist_matrix.index),
                             columns=['x', 'y'])
    df_coords.index.name = 'index'

    # DataFrame of plotted distances
    df_plot_dist = _coord_distances(df_coords, df_dist_matrix)
    df_plot_dist.index.name = 'index'

    # Actual vs Plotted distances
    df_dist_delta = df_plot_dist - df_dist_matrix

    # Combine all 3 into one long table
    df_dist_melted = _coord_melt_distances(df_orig_dist=df_dist_matrix,
                                           df_plot_dist=df_plot_dist,
                                           df_dist_delta=df_dist_delta)

    # Return coords and distances
    return df_coords, df_dist_melted


def plot_anchor_distortion(df_melted: pd.DataFrame) -> Figure:
    """Generate anchor distortion plots.

    Args:
        df (DataFrame): Original and plotted distances (df_melted)

    Returns:
        Tuple(Figure, Figure): Plotly figures (groups, overall)
    """

    # Init
    df = df_melted.copy()
    df = df.reset_index()

    # Get groups (IGHV1 etc.)
    df['group_name'] = [x[:5]
                        for x in list(df['index'])]
    df['group_name2'] = [x[:5]
                         for x in list(df['item2'])]

    is_same_group = df['group_name'] == df['group_name2']

    # Plot distortion within same groups
    df_same_group = df[is_same_group]
    fig1: Figure = px.density_contour(df_same_group, facet_col_wrap=3,
                                      x='orig_distance', y='plot_distance', facet_col='group_name')

    fig1.update_traces(contours_coloring="fill", contours_showlabels=False)
    fig1.update_traces(showscale=False)

    # Plot overall distortion
    fig2: Figure = px.density_contour(
        df, x='orig_distance', y='plot_distance')
    fig2.update_traces(contours_coloring="fill", contours_showlabels=True)

    # Return the figures
    return fig1, fig2


# %% Plot anchors
def plot_anchor_positions(df_coords: pd.DataFrame) -> Figure:
    """Scatter plot of anchor coordinates

    Args:
        df_coords (pd.DataFrame): Anchor coordinates. Index should be the name of the allele.
            First 5 characters will be used for group (e.g. IGHV1)

    Returns:
        Figure: Plotly scatter figure, coloured by group.
    """

    df = df_coords.copy()
    df = df.reset_index()

    # group by IGHV1 etc.
    df['group_name'] = [x[:5] for x in list(df['index'])]
    fig = px.scatter(x=df['x'], y=df['y'],
                     color=df['group_name'])

    return fig

# %%


def db_write_records(df_records: pd.DataFrame, fn_db: str):
    """Create anchor_records table

    Existing table will be overwritten. Creates a unique
    index on anchor_id and anchor_name fields.

    Args:
        df_records (pd.DataFrame): DataFrame containing the anchor records.
        fn_db (str): Database to write to.
    """

    sql_index1 = '''
        CREATE UNIQUE INDEX idx_records_anchor_id
        ON anchor_records(anchor_id)
    '''

    sql_index2 = """
        CREATE UNIQUE INDEX idx_records_anchor_name
        ON anchor_records(anchor_name)
    """

    # Create table and index
    with sqlite3.connect(fn_db) as conn:
        df_records = df_records.copy().reset_index()
        df_records.rename(columns={'index': 'anchor_name'},
                          inplace=True)  # index column is loaded from csv
        # df_records = df_records.convert_dtypes() # Convert from object type
        df_records.to_sql('anchor_records', con=conn, if_exists='replace',
                          index=True, index_label='anchor_id')
        curr = conn.cursor()
        curr.execute(sql_index1)
        curr.execute(sql_index2)
        curr.close()


def db_write_coords(df_coords: pd.DataFrame, fn_db: str):
    """Create database of anchors

    Args:
        df_coords (pd.DataFrame): Anchor coordinates. 'index' column should be the name of the allele.

        fn_db (str): SQLite db to create/write to. 'anchor_records' should exist as anchor_id will
            be pulled from here.
    """

    TABLE_NAME = 'anchor_coords'

    sql_index_anchor_id = f"""
        CREATE UNIQUE INDEX idx_coords_anchor_id
        ON {TABLE_NAME}(anchor_id)
    """

    sql_index_anchor_name = f"""
        CREATE UNIQUE INDEX idx_coords_anchor_name
        ON {TABLE_NAME}(anchor_name)
    """

    # Update anchor_id to be that defined by anchor_records
    sql_update_anchor_id = f"""
        UPDATE {TABLE_NAME}
        SET anchor_id = (SELECT anchor_id
                         FROM anchor_records
                         WHERE anchor_name = anchor_coords.anchor_name)
    """

    with sqlite3.connect(fn_db) as conn:
        curr = conn.cursor()

        # Prepare DataFrame
        df_coords = df_coords.copy()  # Stop changes affecting
        df_coords.insert(0, 'anchor_id', 0)  # will update shortly...
        df_coords.rename(columns={'index': 'anchor_name'},
                         inplace=True)  # index column is loaded from csv

        # Create anchor_coords table
        df_coords.to_sql(name=TABLE_NAME, con=conn, if_exists='replace',
                         index=False)

        # Update anchor_id and index
        curr.execute(sql_update_anchor_id)
        curr.execute(sql_index_anchor_id)
        curr.execute(sql_index_anchor_name)
        curr.close()