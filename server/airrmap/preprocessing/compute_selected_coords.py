# Compute coordinates for a small set of selected sequences.
# Mainly used for the sequence locator markers in the user interface.
# Uses environment configuration.

# %% Imports
import json
from typing import Any, Dict, List
from airrmap.preprocessing.seq_adapter_base import SeqAdapterBase, AnchorItem
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
        These sequences will be mapped to a dictionary/record format using the
        envconfig.application setting for the environment.

    app_cfg : AppConfig (optional)
        Application configuration to use, including the base path for the 
        data. By default, None (use default AppConfig configuration path).

    Returns
    -------
    List
        See SeqAdapterBase.process_single_record().
    """

    # Read environment configuration
    app_cfg = app_cfg if app_cfg is not None else AppConfig()
    env_cfg = app_cfg.get_env_config(env_name)
    cfganc = env_cfg['anchor']
    cfgseq = env_cfg['sequence']
    cfgseq_markers = env_cfg['application']['seq_markers']
    distance_measure_name = env_cfg['distance_measure']
    distance_measure_env_kwargs = env_cfg['distance_measure_env_kwargs']
    distance_measure_seq_kwargs = cfgseq_markers['distance_measure_record_kwargs']
    distance_measure_anchor_kwargs = cfganc['distance_measure_record_kwargs']
    seq_field_delim = cfgseq_markers['field_delim']
    seq_field_mapping = cfgseq_markers['field_mapping']
    sequence_num_closest_anchors = cfgseq['num_closest_anchors']

    # Load anchors
    fn_anchors = app_cfg.get_anchordb(env_name)
    prep_args = SeqAdapterBase.prepare(
        fn=None,
        seq_row_start=0,  # not used
        fn_anchors=fn_anchors
    )
    d_anchors: Dict[int, AnchorItem] = prep_args['anchors']

    # Compute the sequence coordinates
    result_list = []
    for seq in seq_list:
        result = get_single_coords(
            seq=seq,
            seq_field_delim=seq_field_delim,
            seq_field_mapping=seq_field_mapping,
            anchors=d_anchors,
            num_closest_anchors=sequence_num_closest_anchors,
            distance_measure_name=distance_measure_name,
            distance_measure_env_kwargs=distance_measure_env_kwargs,
            distance_measure_seq_kwargs=distance_measure_seq_kwargs,
            distance_measure_anchor_kwargs=distance_measure_anchor_kwargs
        )
        result_list.append(result)

    return result_list


# %%
def get_single_coords(
        seq: Any,
        seq_field_delim: str,
        seq_field_mapping: List[str],
        anchors: Dict[int, AnchorItem],
        num_closest_anchors: int,
        distance_measure_name: str,
        distance_measure_env_kwargs: Dict,
        distance_measure_seq_kwargs: Dict,
        distance_measure_anchor_kwargs: Dict) -> Dict[str, Any]:

    # Split the sequence value to fields
    # and place in dictionary using field mapping from envconfig.
    seq_split = seq.split(seq_field_delim)
    seq_record = {k: v for k, v in zip(seq_field_mapping, seq_split)}

    result: Dict[str, Any] = SeqAdapterBase.process_single_record(
        row=seq_record,
        anchors=anchors,
        num_closest_anchors=num_closest_anchors,
        distance_measure_name=distance_measure_name,
        distance_measure_env_kwargs=distance_measure_env_kwargs,
        distance_measure_seq_kwargs=distance_measure_seq_kwargs,
        distance_measure_anchor_kwargs=distance_measure_anchor_kwargs,
        save_anchor_dists=False
    )

    return result
