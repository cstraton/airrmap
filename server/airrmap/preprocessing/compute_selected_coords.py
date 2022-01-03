# Compute coordinates for a small set of selected sequences.
# Mainly used for the sequence locator markers in the user interface.
# Uses environment configuration.

# %% Imports
import json
from typing import Any, Dict, List
from airrmap.preprocessing.oas_adapter_base import OASAdapterBase, AnchorItem
from airrmap.application.config import AppConfig


# %%
def get_coords(env_name: str, seq_list: List[Any], app_cfg: AppConfig = None):
    """
    Computes the coordinates for a small list of sequences on the fly.

    The list of sequences should be in the same format that the anchors
    in the environment are trained on. This process reads in the
    environment configuration file and the pre-processed anchors.
    In then aligns the sequences based on the environment configuration.

    Parameters
    ----------
    env_name : str
        The name of the environment.

    seq_list : List[Any]
        The list of sequences to compute the sequences for.
        Note that these should match the sequences used for the anchors.
        e.g. If the anchors are CDR3, then CDR3 sequences should be provided.

    app_cfg : AppConfig (optional)
        Application configuration to use, including the base path for the 
        data. By default, None (use default AppConfig configuration path).

    Returns
    -------
    List
        See OASAdapterBase.process_single_record().
    """

    # Read environment configuration
    app_cfg = app_cfg if app_cfg is not None else AppConfig()
    env_cfg = app_cfg.get_env_config(env_name)
    cfganc = env_cfg['anchor']
    cfgseq = env_cfg['sequence']
    distance_measure_name = env_cfg['distance_measure']
    distance_measure_kwargs = env_cfg['distance_measure_options']
    anchor_seq_field = cfganc['seq_field']
    sequence_num_closest_anchors = cfgseq['num_closest_anchors']
    

    # Load anchors
    fn_anchors = app_cfg.get_anchordb(env_name)
    prep_args = OASAdapterBase.prepare(
        fn=None,
        seq_row_start=0,  # not used
        fn_anchors=fn_anchors,
        anchor_seq_field=anchor_seq_field
    )
    d_anchors: Dict[int, AnchorItem] = prep_args['anchors']

    # Compute the sequence coordinates
    result_list = []
    for seq in seq_list:
        result = get_single_coords(
            seq=seq,
            anchors=d_anchors,
            num_closest_anchors=sequence_num_closest_anchors,
            distance_measure_name=distance_measure_name,
            distance_measure_kwargs=distance_measure_kwargs
        )
        result_list.append(result)

    return result_list


# %%
def get_single_coords(
        seq: Any,
        anchors: Dict[int, AnchorItem],
        num_closest_anchors: int,
        distance_measure_name: str,
        distance_measure_kwargs: Dict) -> Dict[str, Any]:

    # Wrap the sequence in a dictionary
    dummy_row = dict(seq=seq)

    result: Dict[str, Any] = OASAdapterBase.process_single_record(
        row=dummy_row,
        seq_field='seq',  # from above dummy_row, not related to field in data files
        anchors=anchors,
        num_closest_anchors=num_closest_anchors,
        distance_measure_name=distance_measure_name,
        distance_measure_kwargs=distance_measure_kwargs,
        save_anchor_dists=False
    )

    return result
